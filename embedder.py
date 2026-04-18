

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ColPaliEmbedder:
    _loaded_resources: ClassVar[Dict[Tuple[str, str], Tuple[Any, Any]]] = {}
    _resource_locks: ClassVar[Dict[Tuple[str, str], threading.Lock]] = {}
    _resolved_model_paths: ClassVar[Dict[str, str]] = {}
    _resolved_model_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str = "vidore/colSmol-256M", device: str = "cpu", pool_factor: int = 1):
        self.model_name = model_name
        self.device     = device
        self.pool_factor = max(1, pool_factor)
        self.dtype      = torch.float32 if device == "cpu" else torch.bfloat16
        self._token_pooler = None
        self._load_model()

    def _get_token_pooler(self):
        if self.pool_factor <= 1:
            return None
        if self._token_pooler is None:
            from colpali_engine.compression.token_pooling import HierarchicalTokenPooler  # type: ignore

            self._token_pooler = HierarchicalTokenPooler()
        return self._token_pooler

    @classmethod
    def _get_resource_lock(cls, key: Tuple[str, str]) -> threading.Lock:
        lock = cls._resource_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            cls._resource_locks[key] = lock
        return lock

   
    def _load_model(self):
        cache_key = (self.model_name, self.device)
        cached_resources = self._loaded_resources.get(cache_key)
        if cached_resources is not None:
            self.processor, self.model = cached_resources
            logger.info("Reusing loaded model %s on %s.", self.model_name, self.device)
            return

        with self._get_resource_lock(cache_key):
            cached_resources = self._loaded_resources.get(cache_key)
            if cached_resources is not None:
                self.processor, self.model = cached_resources
                logger.info("Reusing loaded model %s on %s.", self.model_name, self.device)
                return

            logger.info("Loading %s on %s …", self.model_name, self.device)

            name = self.model_name.lower()

            if "colsmol" in name or "idefics" in name:
                # ColSmol / ColIdefics3 family
                from colpali_engine.models import ColIdefics3, ColIdefics3Processor  # type: ignore
                self.processor = ColIdefics3Processor.from_pretrained(self.model_name)
                self.model = ColIdefics3.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    attn_implementation="eager",
                ).eval()

            elif "colqwen" in name:
                # ColQwen2 family
                from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore
                self.processor = ColQwen2Processor.from_pretrained(self.model_name)
                self.model = ColQwen2.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    attn_implementation="eager",
                ).eval()

            else:
                # ColPali family
                from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
                self.processor = ColPaliProcessor.from_pretrained(self.model_name)
                self.model = ColPali.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    attn_implementation="eager",
                ).eval()

            if self.device != "cpu":
                self.model = self.model.to(self.device)

            self._loaded_resources[cache_key] = (self.processor, self.model)
            logger.info("Model ready.")



    def embed_images(
        self,
        images: List[Image.Image],
        batch_size: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[torch.Tensor]:
        """
        Encode PDF page images → list of float32 tensors.
        Shape per tensor: (num_patches, embedding_dim=128).
        """
        all_embeddings: List[torch.Tensor] = []

        total_batches = max(1, (len(images) + batch_size - 1) // batch_size)
        if progress_callback is not None:
            progress_callback(0, total_batches)

        for batch_index, i in enumerate(range(0, len(images), batch_size), start=1):
            batch = images[i : i + batch_size]
            with torch.no_grad():
                inputs     = self.processor.process_images(batch).to(self.device)
                embeddings = self.model(**inputs)
            if self.pool_factor > 1:
                token_pooler = self._get_token_pooler()
                embeddings = token_pooler.pool_embeddings(
                    list(torch.unbind(embeddings)),
                    pool_factor=self.pool_factor,
                )
            for emb in embeddings:
                all_embeddings.append(emb.cpu().float())   # float32 for Qdrant
            if progress_callback is not None:
                progress_callback(batch_index, total_batches)

        logger.info("Embedded %d pages.", len(all_embeddings))
        return all_embeddings

   
    def embed_query(self, query: str) -> torch.Tensor:
        """Encode a text query → float32 tensor (num_tokens, 128)."""
        with torch.no_grad():
            inputs    = self.processor.process_queries([query]).to(self.device)
            embedding = self.model(**inputs)
        return embedding[0].cpu().float()

  
    @staticmethod
    def maxsim_score(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        q = torch.nn.functional.normalize(query_emb, dim=-1)
        d = torch.nn.functional.normalize(doc_emb, dim=-1)
        return float(torch.matmul(q, d.T).max(dim=-1).values.sum().item())
