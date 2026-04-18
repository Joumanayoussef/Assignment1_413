from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from statistics import median
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_pdf_bytes(source: bytes | str | Path) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            return path.read_bytes()

        import urllib.request

        logger.info("Downloading PDF from %s …", source)
        with urllib.request.urlopen(str(source), timeout=60) as response:
            return response.read()

    return source


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_block_text(block: dict) -> str:
    lines = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = _normalize_text(" ".join(spans))
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def _collect_font_sizes(text_dict: dict) -> List[float]:
    sizes: List[float] = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size")
                if isinstance(size, (int, float)):
                    sizes.append(float(size))
    return sizes


def _is_heading(text: str, max_font_size: float, baseline_font_size: float) -> bool:
    compact = _normalize_text(text)
    if not compact or len(compact) > 120:
        return False
    title_like = compact.istitle() or compact.isupper() or compact.endswith(":")
    return max_font_size >= baseline_font_size + 1.0 or title_like


def _looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    numeric_lines = sum(bool(re.search(r"\d", line)) for line in lines)
    columnish_lines = sum(line.count("  ") >= 2 or line.count("|") >= 2 for line in lines)
    return numeric_lines >= 2 and columnish_lines >= 1


def _looks_like_chart_caption(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("figure", "chart", "graph", "diagram", "plot", "visual"))


def _chunk_text(text: str, max_chars: int = 650) -> List[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)
    return chunks


def pdf_to_images_poppler(pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    """Convert every page of a PDF to a PIL RGB Image using pdf2image / Poppler."""
    from pdf2image import convert_from_bytes  # type: ignore

    images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="jpeg")
    logger.info("pdf2image: converted %d pages at %d DPI", len(images), dpi)
    return images


def pdf_to_images_pymupdf(pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    """Convert every page of a PDF to a PIL RGB Image using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "Neither pdf2image+Poppler nor PyMuPDF is available. "
            "Install with: pip install pymupdf"
        )

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    logger.info("PyMuPDF: converted %d pages at %d DPI", len(images), dpi)
    return images


def load_pdf(source: bytes | str | Path, dpi: int = 150) -> List[Image.Image]:
    """Load a PDF and return a list of PIL Images, one per page."""
    pdf_bytes = _resolve_pdf_bytes(source)

    try:
        return pdf_to_images_poppler(pdf_bytes, dpi=dpi)
    except Exception as e_pop:
        logger.warning("pdf2image failed (%s); falling back to PyMuPDF.", e_pop)
        return pdf_to_images_pymupdf(pdf_bytes, dpi=dpi)


def extract_pdf_page_texts(source: bytes | str | Path) -> List[str]:
    """Extract text from each PDF page using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for grounded text answers. Install with: pip install pymupdf"
        ) from exc

    pdf_bytes = _resolve_pdf_bytes(source)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts = [_normalize_text(page.get_text("text")) for page in doc]
    logger.info("PyMuPDF: extracted text from %d pages", len(page_texts))
    return page_texts


def extract_pdf_structure(source: bytes | str | Path) -> List[dict]:
    """Extract per-page text, structural chunks, table data, and visual metadata."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for structured PDF extraction. Install with: pip install pymupdf"
        ) from exc

    pdf_bytes = _resolve_pdf_bytes(source)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[dict] = []

    for page_number, page in enumerate(doc, start=1):
        text_dict = page.get_text("dict")
        raw_text = _normalize_text(page.get_text("text"))
        font_sizes = _collect_font_sizes(text_dict)
        baseline_font_size = median(font_sizes) if font_sizes else 11.0
        section_title = f"Page {page_number}"
        section_titles: List[str] = []
        chunks: List[dict] = []
        table_count = 0
        chart_count = 0
        text_count = 0

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            block_text = _extract_block_text(block)
            if not block_text:
                continue

            span_sizes = [
                float(span.get("size", baseline_font_size))
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ]
            max_font_size = max(span_sizes, default=baseline_font_size)

            if _is_heading(block_text, max_font_size, baseline_font_size):
                section_title = _normalize_text(block_text)[:120]
                section_titles.append(section_title)
                continue

            if _looks_like_table(block_text):
                chunk_type = "table"
                table_count += 1
            elif _looks_like_chart_caption(block_text):
                chunk_type = "chart"
                chart_count += 1
            else:
                chunk_type = "text"
                text_count += 1

            max_chars = 700 if chunk_type == "table" else 550
            for chunk_text in _chunk_text(block_text, max_chars=max_chars):
                chunks.append(
                    {
                        "section_title": section_title,
                        "chunk_type": chunk_type,
                        "text": chunk_text,
                    }
                )

        if hasattr(page, "find_tables"):
            try:
                tables = page.find_tables()
                table_items = getattr(tables, "tables", tables)
                for index, table in enumerate(table_items, start=1):
                    extracted_rows = table.extract() if hasattr(table, "extract") else []
                    formatted_rows = [
                        " | ".join(_normalize_text(cell or "") for cell in row)
                        for row in extracted_rows
                        if any((cell or "").strip() for cell in row)
                    ]
                    if not formatted_rows:
                        continue
                    table_count += 1
                    chunks.append(
                        {
                            "section_title": f"Table {index}",
                            "chunk_type": "table",
                            "text": "\n".join(formatted_rows),
                        }
                    )
            except Exception as exc:
                logger.debug("Table extraction skipped on page %d: %s", page_number, exc)

        image_entries = page.get_images(full=True)
        image_count = len(image_entries)
        drawing_count = len(page.get_drawings())

        if image_count:
            chunks.append(
                {
                    "section_title": f"Visual assets on page {page_number}",
                    "chunk_type": "image",
                    "text": f"This page contains {image_count} embedded image(s).",
                }
            )

        if drawing_count >= 8:
            chart_count += 1
            chunks.append(
                {
                    "section_title": f"Chart or diagram on page {page_number}",
                    "chunk_type": "chart",
                    "text": f"This page contains {drawing_count} vector drawing elements, which suggests a chart, diagram, or structured figure.",
                }
            )

        for index, chunk in enumerate(chunks, start=1):
            chunk["chunk_id"] = f"p{page_number}-c{index}"
            chunk["page_number"] = page_number

        pages.append(
            {
                "page_number": page_number,
                "page_text": raw_text,
                "chunks": chunks,
                "modality_summary": {
                    "text_chunks": text_count,
                    "table_chunks": table_count,
                    "chart_chunks": chart_count,
                    "image_count": image_count,
                    "drawing_count": drawing_count,
                    "chunk_count": len(chunks),
                },
                "section_titles": section_titles,
            }
        )

    logger.info("PyMuPDF: extracted structured content from %d pages", len(pages))
    return pages


def describe_pages(images: List[Image.Image]) -> List[dict]:
    """Return lightweight metadata for each page image."""
    return [
        {
            "page": i + 1,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
        }
        for i, img in enumerate(images)
    ]


def image_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> bytes:
    """Serialize a PIL Image to bytes (for storage / display)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()
