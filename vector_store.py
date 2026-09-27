from __future__ import annotations

import io
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "about", "into", "than",
    "their", "there", "these", "those", "were", "will", "would", "could", "should",
}


def _tensor_to_list(t: torch.Tensor) -> List[List[float]]:
    """Convert a 2-D tensor to a list-of-lists (Qdrant multi-vector format)."""
    return t.numpy().tolist()


def _image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in STOPWORDS}


def _phrase_score(query_text: str, candidate_text: str) -> float:
    query_terms = [term for term in re.findall(r"[A-Za-z0-9]+", query_text.lower()) if term not in STOPWORDS]
    candidate_lower = candidate_text.lower()
    bonus = 0.0
    for term in query_terms:
        if term in candidate_lower:
            bonus += 0.4
    if len(query_terms) >= 2:
        bigrams = [" ".join(query_terms[i:i + 2]) for i in range(len(query_terms) - 1)]
        for bigram in bigrams:
            if bigram in candidate_lower:
                bonus += 1.2
    return bonus


def _coerce_multivector(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, dict):
        for candidate in value.values():
            tensor = _coerce_multivector(candidate)
            if tensor is not None:
                return tensor
        return None
    try:
        tensor = torch.tensor(value, dtype=torch.float32)
    except Exception:
        return None
    if tensor.ndim != 2:
        return None
    return tensor


def _avg_maxsim_score(query_emb: torch.Tensor, doc_emb: Optional[torch.Tensor]) -> Optional[float]:
    if doc_emb is None or doc_emb.numel() == 0:
        return None
    q = torch.nn.functional.normalize(query_emb, dim=-1)
    d = torch.nn.functional.normalize(doc_emb, dim=-1)
    return float(torch.matmul(q, d.T).max(dim=-1).values.mean().item())


def _rank_chunks(query_text: str, chunks: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    if not query_text:
        return chunks[:limit]

    query_tokens = _tokenize(query_text)
    lowered_query = query_text.lower()
    ranked: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk_text = f"{chunk.get('section_title', '')} {chunk.get('text', '')}"
        chunk_tokens = _tokenize(chunk_text)
        overlap = len(query_tokens & chunk_tokens)
        overlap_ratio = overlap / max(1, len(query_tokens))
        chunk_type = chunk.get("chunk_type", "text")
        modality_bonus = 0.0
        if any(term in lowered_query for term in ("table", "row", "column", "numeric")) and chunk_type == "table":
            modality_bonus += 2.0
        if any(term in lowered_query for term in ("chart", "graph", "figure", "diagram", "plot")) and chunk_type in {"chart", "image"}:
            modality_bonus += 2.0
        if any(term in lowered_query for term in ("image", "photo", "visual")) and chunk_type == "image":
            modality_bonus += 1.5

        phrase_bonus = _phrase_score(query_text, chunk_text)
        section_bonus = _phrase_score(query_text, chunk.get("section_title", ""))
        score = (2.5 * overlap_ratio) + overlap + modality_bonus + phrase_bonus + (0.5 * section_bonus)
        if score <= 0 and query_tokens:
            continue

        ranked_chunk = dict(chunk)
        ranked_chunk["match_score"] = score
        ranked.append(ranked_chunk)

    ranked.sort(key=lambda chunk: chunk.get("match_score", 0.0), reverse=True)
    return ranked[:limit]


class VectorStore:
    VECTOR_NAME = "colpali"

    def __init__(
        self,
        collection_name: str = "colpali_docs",
        mode: str = "local",
        url: str = "",
        api_key: str = "",
        embedding_dim: int = 128,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        if mode == "cloud" and url:
            self.client = QdrantClient(url=url, api_key=api_key or None)
            logger.info("Connected to Qdrant Cloud at %s", url)
        else:
            self.client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant (local mode).")

        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it doesn't exist yet."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.VECTOR_NAME: qmodels.VectorParams(
                        size=self.embedding_dim,
                        distance=qmodels.Distance.COSINE,
                        multivector_config=qmodels.MultiVectorConfig(
                            comparator=qmodels.MultiVectorComparator.MAX_SIM
                        ),
                    )
                },
            )
            logger.info("Created Qdrant collection '%s'.", self.collection_name)

    def reset_collection(self):
        """Delete and recreate the collection (clear all documents)."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.info("Collection '%s' reset.", self.collection_name)

    def index_pages(
        self,
        doc_id: str,
        doc_name: str,
        page_images: List[Image.Image],
        page_embeddings: List[torch.Tensor],
        page_data: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Store all pages of a document in Qdrant."""
        points: List[qmodels.PointStruct] = []
        page_data = page_data or [{} for _ in page_images]

        for page_idx, (img, emb, page_info) in enumerate(zip(page_images, page_embeddings, page_data)):
            point_id = str(uuid.uuid4())
            img_bytes = _image_to_bytes(img)

            point = qmodels.PointStruct(
                id=point_id,
                vector={self.VECTOR_NAME: _tensor_to_list(emb)},
                payload={
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_number": page_idx + 1,
                    "total_pages": len(page_images),
                    "page_text": page_info.get("page_text", ""),
                    "chunks": page_info.get("chunks", []),
                    "modality_summary": page_info.get("modality_summary", {}),
                    "section_titles": page_info.get("section_titles", []),
                    "image_bytes": img_bytes.hex(),
                },
            )
            points.append(point)

        batch_size = 32
        total_batches = max(1, (len(points) + batch_size - 1) // batch_size)
        for batch_index, i in enumerate(range(0, len(points), batch_size), start=1):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )
            if progress_callback is not None:
                progress_callback(batch_index, total_batches)

        logger.info("Indexed %d pages for doc '%s'.", len(points), doc_name)
        return len(points)

    def search(
        self,
        query_embedding: torch.Tensor,
        query_text: str = "",
        top_k: int = 3,
        doc_id: Optional[str] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant pages for a query embedding."""
        query_vectors = _tensor_to_list(query_embedding)

        search_filter = None
        if doc_id:
            search_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=doc_id),
                    )
                ]
            )

        candidate_limit = max(top_k * 4, top_k)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vectors,
            using=self.VECTOR_NAME,
            limit=candidate_limit,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=True,
        )

        hits = []
        for point in results.points:
            payload = point.payload or {}
            img_bytes = bytes.fromhex(payload.get("image_bytes", ""))
            img = Image.open(io.BytesIO(img_bytes)) if img_bytes else None
            ranked_chunks = _rank_chunks(query_text, payload.get("chunks", []), limit=4)
            best_chunk_score = ranked_chunks[0].get("match_score", 0.0) if ranked_chunks else 0.0
            exact_score = _avg_maxsim_score(query_embedding, _coerce_multivector(getattr(point, "vector", None)))
            hybrid_score = (exact_score if exact_score is not None else float(point.score)) + (0.08 * best_chunk_score)
            hits.append(
                {
                    "score": point.score,
                    "exact_score": exact_score,
                    "hybrid_score": hybrid_score,
                    "page_number": payload.get("page_number"),
                    "doc_id": payload.get("doc_id"),
                    "doc_name": payload.get("doc_name"),
                    "total_pages": payload.get("total_pages"),
                    "page_text": payload.get("page_text", ""),
                    "chunks": ranked_chunks,
                    "modality_summary": payload.get("modality_summary", {}),
                    "section_titles": payload.get("section_titles", []),
                    "image": img,
                }
            )

        if rerank and query_text and len(hits) > 1:
            from semantic_reranker import SemanticReranker
            page_texts = [hit.get("page_text", "") or "" for hit in hits]
            sem_scores = SemanticReranker.score(query_text, page_texts)
            for hit, sem_score in zip(hits, sem_scores):
                hit["semantic_score"] = round(sem_score, 4)
                hit["hybrid_score"] = hit["hybrid_score"] + 0.15 * sem_score

        hits.sort(key=lambda hit: hit.get("hybrid_score", hit.get("score", 0.0)), reverse=True)
        return hits[:top_k]

    def count(self) -> int:
        """Return total number of indexed page points."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return unique documents in the collection."""
        scroll_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=1000,
        )
        seen: Dict[str, dict] = {}
        for point in scroll_result:
            payload = point.payload or {}
            doc_id = payload.get("doc_id", "")
            modality_summary = payload.get("modality_summary", {})
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": payload.get("doc_name", "Unknown"),
                    "total_pages": payload.get("total_pages", 0),
                    "chunk_count": 0,
                    "table_count": 0,
                    "chart_count": 0,
                    "image_count": 0,
                }
            seen[doc_id]["chunk_count"] += int(modality_summary.get("chunk_count", 0))
            seen[doc_id]["table_count"] += int(modality_summary.get("table_chunks", 0))
            seen[doc_id]["chart_count"] += int(modality_summary.get("chart_chunks", 0))
            seen[doc_id]["image_count"] += int(modality_summary.get("image_count", 0))
        return list(seen.values())
