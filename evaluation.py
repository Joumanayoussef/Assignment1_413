from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


BENCHMARK_QUERIES = [
    {
        "query": "Summarize the main findings or conclusions in this document.",
        "keywords": ["summary", "conclusion", "finding", "result"],
        "modality": "text",
        "expected_modalities": ["text"],
    },
    {
        "query": "What tables, numeric comparisons, or structured data appear in the document?",
        "keywords": ["table", "column", "row", "data", "comparison"],
        "modality": "table",
        "expected_modalities": ["table"],
    },
    {
        "query": "Describe any figures, charts, diagrams, or visual elements that matter.",
        "keywords": ["figure", "chart", "graph", "diagram", "visual"],
        "modality": "chart",
        "expected_modalities": ["chart", "image"],
    },
    {
        "query": "Which section best answers the main research question or topic?",
        "keywords": ["section", "topic", "research", "question"],
        "modality": "section-aware",
        "expected_modalities": ["text", "table", "chart"],
    },
    {
        "query": "What evidence is provided across text, tables, and visuals for the topic being discussed?",
        "keywords": ["evidence", "text", "table", "visual", "support"],
        "modality": "multi-modal",
        "expected_modalities": ["text", "table", "chart", "image"],
    },
]


@dataclass
class BenchmarkResult:
    query: str
    modality: str
    top_pages: List[int]
    top_docs: List[str]
    scores: List[float]
    top_sections: List[str]
    answer: str
    latency_s: float
    keyword_hit: bool
    citation_count: int
    modality_hit: bool
    evidence_count: int
    keywords_checked: List[str]


class Evaluator:
    """Runs benchmark queries and collects retrieval and grounding metrics."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 3,
        doc_id: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        queries = queries or BENCHMARK_QUERIES
        results: List[BenchmarkResult] = []

        for item in queries:
            query = item["query"]
            keywords = item.get("keywords", [])
            modality = item.get("modality", "text")
            expected_modalities = set(item.get("expected_modalities", []))

            started_at = time.time()
            out = self.pipeline.query(query, top_k=top_k, doc_id=doc_id)
            latency = time.time() - started_at

            answer = out.get("answer", "")
            hits = out.get("hits", [])
            citations = out.get("citations", [])
            evidence = out.get("evidence", [])

            keyword_hit = any(keyword.lower() in answer.lower() for keyword in keywords)
            citation_modalities = {citation.get("chunk_type", "text") for citation in citations}
            modality_hit = bool(citation_modalities & expected_modalities) if expected_modalities else bool(citations)

            results.append(
                BenchmarkResult(
                    query=query,
                    modality=modality,
                    top_pages=[hit.get("page_number", 0) for hit in hits],
                    top_docs=[hit.get("doc_name", "") for hit in hits],
                    scores=[round(hit.get("hybrid_score", hit.get("score", 0.0)), 4) for hit in hits],
                    top_sections=[citation.get("section_title", "") for citation in citations],
                    answer=answer,
                    latency_s=round(latency, 2),
                    keyword_hit=keyword_hit,
                    citation_count=len(citations),
                    modality_hit=modality_hit,
                    evidence_count=len(evidence),
                    keywords_checked=keywords,
                )
            )

        return results

    @staticmethod
    def summary(results: List[BenchmarkResult]) -> Dict[str, Any]:
        if not results:
            return {}

        total = len(results)
        avg_latency = sum(result.latency_s for result in results) / total
        keyword_hit_rate = sum(result.keyword_hit for result in results) / total
        citation_rate = sum(result.citation_count > 0 for result in results) / total
        modality_hit_rate = sum(result.modality_hit for result in results) / total
        avg_evidence_count = sum(result.evidence_count for result in results) / total
        avg_top_score = sum(result.scores[0] if result.scores else 0.0 for result in results) / total
        modality_counts: Dict[str, int] = {}
        for result in results:
            modality_counts[result.modality] = modality_counts.get(result.modality, 0) + 1

        return {
            "total_queries": total,
            "avg_latency_s": round(avg_latency, 2),
            "keyword_hit_rate": round(keyword_hit_rate * 100, 1),
            "citation_coverage_rate": round(citation_rate * 100, 1),
            "modality_hit_rate": round(modality_hit_rate * 100, 1),
            "avg_evidence_count": round(avg_evidence_count, 2),
            "avg_top_retrieval_score": round(avg_top_score, 4),
            "modality_breakdown": modality_counts,
        }
