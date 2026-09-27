from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from config import cfg
from embedder import ColPaliEmbedder
from generator import MockGenerator, OpenAIGenerator
from ingestion import extract_pdf_structure, load_pdf
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end Multi-Modal RAG pipeline."""

    def __init__(self):
        self._embedder: Optional[ColPaliEmbedder] = None
        self._vector_store: Optional[VectorStore] = None
        self._generator = None
        self._generator_signature: Optional[tuple[str, str]] = None

    @property
    def embedder(self) -> ColPaliEmbedder:
        if self._embedder is None:
            self._embedder = ColPaliEmbedder(
                model_name=cfg.colpali.model_name,
                device=cfg.colpali.device,
                pool_factor=cfg.colpali.pool_factor,
            )
        elif (
            self._embedder.model_name != cfg.colpali.model_name
            or self._embedder.device != cfg.colpali.device
            or self._embedder.pool_factor != cfg.colpali.pool_factor
        ):
            logger.info(
                "Reloading embedder with model=%s device=%s",
                cfg.colpali.model_name,
                cfg.colpali.device,
            )
            self._embedder = ColPaliEmbedder(
                model_name=cfg.colpali.model_name,
                device=cfg.colpali.device,
                pool_factor=cfg.colpali.pool_factor,
            )
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                collection_name=cfg.qdrant.collection_name,
                mode=cfg.qdrant.mode,
                url=cfg.qdrant.url,
                api_key=cfg.qdrant.api_key,
                embedding_dim=cfg.qdrant.embedding_dim,
            )
        return self._vector_store

    @property
    def generator(self):
        signature = (cfg.openai.api_key.strip(), cfg.openai.model)
        if self._generator is None or self._generator_signature != signature:
            if cfg.openai.api_key.strip():
                self._generator = OpenAIGenerator(
                    api_key=cfg.openai.api_key.strip(),
                    model=cfg.openai.model,
                    max_tokens=cfg.openai.max_tokens,
                )
            else:
                self._generator = MockGenerator()
            self._generator_signature = signature
        return self._generator

    def ingest_pdf(
        self,
        source,
        doc_name: str = "document.pdf",
        doc_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline for one PDF.
        progress_callback(step, total, message, progress) is called at each stage.
        """
        doc_id = doc_id or str(uuid.uuid4())
        cb = progress_callback or (lambda s, t, m, p=None: None)

        cb(1, 5, "Reading PDF, converting pages, and extracting structure...", 0.05)
        page_images: List[Image.Image] = load_pdf(source, dpi=cfg.pdf_dpi)
        page_data = extract_pdf_structure(source)
        if len(page_data) != len(page_images):
            logger.warning(
                "Page image count (%d) and page data count (%d) differ; aligning to the shorter length.",
                len(page_images),
                len(page_data),
            )
            aligned_count = min(len(page_images), len(page_data))
            page_images = page_images[:aligned_count]
            page_data = page_data[:aligned_count]

        total_chunks = sum(page.get("modality_summary", {}).get("chunk_count", 0) for page in page_data)
        logger.info("'%s' → %d pages, %d chunks", doc_name, len(page_images), total_chunks)
        cb(1, 5, f"Converted PDF into {len(page_images)} pages and extracted {total_chunks} structural chunks.", 1.0)

        cb(2, 5, f"Loading embedding model: {cfg.colpali.model_name}", 0.1)
        embedder = self.embedder
        cb(2, 5, f"Model ready on {cfg.colpali.device}.", 1.0)

        def on_embedding_progress(done: int, total: int):
            cb(
                3,
                5,
                f"Embedding page batch {done}/{total} ({len(page_images)} pages total)...",
                max(0.05, done / total),
            )

        cb(3, 5, f"Embedding {len(page_images)} pages with {cfg.colpali.model_name}...", 0.05)
        page_embeddings = embedder.embed_images(page_images, progress_callback=on_embedding_progress)
        cb(3, 5, f"Created embeddings for {len(page_embeddings)} pages.", 1.0)

        def on_index_progress(done: int, total: int):
            cb(
                4,
                5,
                f"Saving batches to the vector store {done}/{total}...",
                done / total,
            )

        cb(4, 5, "Saving page embeddings and structural chunks into the vector store...", 0.0)
        n_indexed = self.vector_store.index_pages(
            doc_id=doc_id,
            doc_name=doc_name,
            page_images=page_images,
            page_embeddings=page_embeddings,
            page_data=page_data,
            progress_callback=on_index_progress,
        )
        cb(4, 5, f"Stored {n_indexed} pages in the vector store.", 1.0)

        totals = {
            "chunks": total_chunks,
            "tables": sum(page.get("modality_summary", {}).get("table_chunks", 0) for page in page_data),
            "charts": sum(page.get("modality_summary", {}).get("chart_chunks", 0) for page in page_data),
            "images": sum(page.get("modality_summary", {}).get("image_count", 0) for page in page_data),
        }

        cb(5, 5, f"Done. Indexed {n_indexed} pages successfully.", 1.0)
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "num_pages": len(page_images),
            "num_indexed": n_indexed,
            "num_chunks": totals["chunks"],
            "num_tables": totals["tables"],
            "num_charts": totals["charts"],
            "num_images": totals["images"],
        }

    def query(
        self,
        question: str,
        top_k: int = 3,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant pages and generate a grounded answer."""
        if self.vector_store.count() == 0:
            return {
                "answer": "⚠️ No documents indexed yet. Please upload a PDF first.",
                "citations": [],
                "hits": [],
                "model": "none",
            }

        query_emb = self.embedder.embed_query(question)
        hits = self.vector_store.search(
            query_embedding=query_emb,
            query_text=question,
            top_k=top_k,
            doc_id=doc_id,
            rerank=cfg.reranker_enabled,
        )
        result = self.generator.generate_answer(question=question, retrieved_hits=hits)
        result["hits"] = hits
        return result

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.vector_store.list_documents()

    def reset(self):
        self.vector_store.reset_collection()
        logger.info("Vector store reset.")
