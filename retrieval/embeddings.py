import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingExtractor:
    """
    Extracts dense vector embeddings for RAG retrieval index.
    Uses SentenceTransformers ('all-MiniLM-L6-v2' or specified model).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype("float32")
