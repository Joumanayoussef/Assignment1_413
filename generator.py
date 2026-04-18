from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "about", "into", "than",
    "their", "there", "these", "those", "were", "will", "would", "could", "should",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _truncate(text: str, max_chars: int = 220) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _looks_like_noise(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    lowered = normalized.lower()

    if not normalized or len(normalized) < 25:
        return True
    if lowered.startswith("contents") or "table of contents" in lowered:
        return True
    if re.search(r"\.{4,}", normalized):
        return True
    if re.fullmatch(r"[\d\s.]+", normalized):
        return True
    if normalized.count("| ") >= 4:
        return True
    if len(re.findall(r"\b\d+(?:\.\d+)?\b", normalized)) >= 6 and len(normalized) < 180:
        return True
    return False


def _clean_sentence(text: str) -> str:
    text = _normalize_whitespace(text)
    text = re.sub(r"^[\-•*\d.)\s]+", "", text)
    return text.strip()


def _question_type(question: str) -> str:
    lowered = question.lower()
    if any(term in lowered for term in ("summary", "summarize", "main idea", "overview", "conclusion")):
        return "summary"
    if any(term in lowered for term in ("why", "how", "explain")):
        return "explanation"
    return "fact"


def _phrase_score(question: str, text: str) -> float:
    tokens = _tokenize(question)
    text_lower = text.lower()
    bonus = 0.0
    for token in tokens:
        if token in text_lower:
            bonus += 0.4
    if len(tokens) >= 2:
        bigrams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        for bigram in bigrams:
            if bigram in text_lower:
                bonus += 1.0
    return bonus


def _compose_answer(question: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return (
            "I could not find enough clear evidence in the indexed PDF pages to answer this question confidently. "
            "Try asking a more specific question or re-index the document."
        )

    answer_style = _question_type(question)
    top_items = evidence[:3] if answer_style == "summary" else evidence[:2]
    statements = []
    seen = set()

    for item in top_items:
        sentence = _clean_sentence(item["text"])
        key = sentence.lower()
        if not sentence or key in seen:
            continue
        seen.add(key)
        statements.append(sentence)

    if not statements:
        return (
            "I found related pages, but the extracted text was too noisy to form a reliable answer. "
            "Open the retrieved pages and sources below to inspect the exact content."
        )

    if answer_style == "summary":
        return " ".join(statements)

    if answer_style == "explanation" and len(statements) >= 2:
        return f"{statements[0]} {statements[1]}"

    if len(statements) == 1:
        return statements[0]

    return " ".join(statements)


def _image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _rank_evidence(question: str, hits: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    question_tokens = set(_tokenize(question))
    answer_style = _question_type(question)
    ranked: List[Dict[str, Any]] = []

    for hit in hits:
        hit_score = float(hit.get("hybrid_score", hit.get("score", 0.0)))
        chunks = hit.get("chunks") or []
        if chunks:
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                if not chunk_text or _looks_like_noise(chunk_text):
                    continue
                for sentence in _split_sentences(chunk_text):
                    sentence = _clean_sentence(sentence)
                    if _looks_like_noise(sentence):
                        continue
                    token_overlap = len(question_tokens & set(_tokenize(sentence)))
                    phrase_bonus = _phrase_score(question, sentence)
                    section_bonus = _phrase_score(question, chunk.get("section_title", ""))
                    if answer_style != "summary" and token_overlap == 0 and phrase_bonus == 0 and question_tokens:
                        continue
                    ranked.append(
                        {
                            "text": sentence,
                            "page_number": hit.get("page_number"),
                            "doc_name": hit.get("doc_name"),
                            "section_title": chunk.get("section_title") or f"Page {hit.get('page_number')}",
                            "chunk_type": chunk.get("chunk_type", "text"),
                            "score": hit_score + float(chunk.get("match_score", 0.0)) + (1.5 * token_overlap) + phrase_bonus + (0.5 * section_bonus),
                        }
                    )
            continue

        page_text = hit.get("page_text", "")
        for sentence in _split_sentences(page_text):
            sentence = _clean_sentence(sentence)
            if _looks_like_noise(sentence):
                continue
            sentence_tokens = set(_tokenize(sentence))
            overlap = len(question_tokens & sentence_tokens)
            phrase_bonus = _phrase_score(question, sentence)
            if answer_style != "summary" and overlap == 0 and phrase_bonus == 0 and question_tokens:
                continue
            ranked.append(
                {
                    "text": sentence,
                    "page_number": hit.get("page_number"),
                    "doc_name": hit.get("doc_name"),
                    "section_title": f"Page {hit.get('page_number')}",
                    "chunk_type": "text",
                    "score": hit_score + (1.5 * overlap) + phrase_bonus,
                }
            )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in ranked:
        key = _normalize_whitespace(item["text"]).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


class MockGenerator:
    """Builds a grounded local answer from retrieved page text and chunks."""

    def generate_answer(
        self,
        question: str,
        retrieved_hits: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        evidence = _rank_evidence(question, retrieved_hits)
        citations = [
            {
                "page": item["page_number"],
                "doc_name": item["doc_name"],
                "score": round(item["score"], 4),
                "section_title": item["section_title"],
                "chunk_type": item["chunk_type"],
            }
            for item in evidence
        ]
        unique_citations = []
        seen_citations = set()
        for citation in citations:
            key = (citation["page"], citation["doc_name"], citation["section_title"], citation["chunk_type"])
            if key in seen_citations:
                continue
            seen_citations.add(key)
            unique_citations.append(citation)

        if evidence:
            answer = _compose_answer(question, evidence)
        elif retrieved_hits:
            pages_str = ", ".join(
                f"Page {hit.get('page_number')} ({hit.get('doc_name')})" for hit in retrieved_hits
            )
            answer = (
                "**Grounded answer unavailable from extracted text.**\n\n"
                f"The retrieval system found relevant pages: {pages_str}.\n\n"
                "Open the retrieved pages below to inspect the source content directly."
            )
        else:
            answer = (
                "**No grounded answer found.**\n\n"
                "The indexed PDF did not return any relevant pages for this question."
            )

        return {
            "answer": answer,
            "citations": unique_citations,
            "model": "local-grounded",
            "evidence": evidence,
        }


class OpenAIGenerator:
    """Uses an OpenAI multimodal model to answer from retrieved PDF pages while keeping citations separate."""

    SYSTEM_PROMPT = (
        "You answer questions about retrieved PDF pages. Use only the provided page images and extracted evidence. "
        "Write a concise, understandable answer. Do not invent facts. Do not inline citations inside the answer body."
    )

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 900):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self._fallback = MockGenerator()

    def generate_answer(
        self,
        question: str,
        retrieved_hits: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        fallback_result = self._fallback.generate_answer(question=question, retrieved_hits=retrieved_hits)
        evidence = fallback_result.get("evidence", [])
        citations = fallback_result.get("citations", [])

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Question: " + question + "\n\n"
                    "Extracted evidence:\n" +
                    "\n".join(
                        f"- Page {item['page_number']} | {item['doc_name']} | {item['section_title']}: {item['text']}"
                        for item in evidence[:6]
                    )
                ),
            }
        ]

        for hit in retrieved_hits[:3]:
            if hit.get("image") is None:
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(hit["image"]), "detail": "low"},
                }
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content.strip() if response.choices else fallback_result["answer"]
        except Exception as exc:
            logger.warning("OpenAI answer generation failed; falling back to local grounded answer: %s", exc)
            return fallback_result

        return {
            "answer": answer,
            "citations": citations,
            "model": self.model,
            "evidence": evidence,
        }
