from __future__ import annotations

import logging
import threading
from typing import Iterable, List

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class SemanticReranker:
    _model = None
    _tokenizer = None
    _lock = threading.Lock()
    _model_name = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def _load(cls):
        if cls._model is not None and cls._tokenizer is not None:
            return

        with cls._lock:
            if cls._model is not None and cls._tokenizer is not None:
                return
            logger.info("Loading semantic reranker model %s", cls._model_name)
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            cls._model = AutoModel.from_pretrained(cls._model_name).eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @classmethod
    def encode(cls, texts: Iterable[str]) -> torch.Tensor:
        cls._load()
        text_list = list(texts)
        if not text_list:
            return torch.empty((0, 384), dtype=torch.float32)

        encoded = cls._tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = cls._model(**encoded)
        embeddings = cls._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    @classmethod
    def score(cls, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        embeddings = cls.encode([query, *candidates])
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        return torch.matmul(candidate_embeddings, query_embedding).cpu().tolist()