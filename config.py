
import os
import torch
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

_has_cuda = torch.cuda.is_available()
_default_model = os.getenv("COLPALI_MODEL_NAME", "vidore/colSmol-256M")
_default_device = os.getenv("COLPALI_DEVICE")


def resolve_colpali_device(model_name: str) -> str:
    if _default_device:
        return _default_device
    if "colsmol" in model_name.lower() or "idefics" in model_name.lower():
        return "cpu"
    return "cuda" if _has_cuda else "cpu"


@dataclass
class ColPaliConfig:
    # Project default → colSmol-256M (ColIdefics3 backbone, ~500 MB, CPU-friendly)
    # Override with COLPALI_MODEL_NAME / COLPALI_DEVICE if needed.
    model_name: str = _default_model
    device: str    = resolve_colpali_device(_default_model)
    pool_factor: int = int(os.getenv("COLPALI_POOL_FACTOR", "2"))


@dataclass
class QdrantConfig:
    mode: str            = "local"
    url: str             = os.getenv("QDRANT_URL", "")
    api_key: str         = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "colpali_docs")
    embedding_dim: int   = 128


@dataclass
class OpenAIConfig:
    api_key: str    = os.getenv("OPENAI_API_KEY", "")
    model: str      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "900"))


@dataclass
class AppConfig:
    colpali:  ColPaliConfig = field(default_factory=ColPaliConfig)
    qdrant:   QdrantConfig  = field(default_factory=QdrantConfig)
    openai:   OpenAIConfig  = field(default_factory=OpenAIConfig)

    top_k:            int  = 3
    pdf_dpi:          int  = 72    # 72 = fast (CPU), use 150 for better quality on GPU
    reranker_enabled: bool = True


cfg = AppConfig()
