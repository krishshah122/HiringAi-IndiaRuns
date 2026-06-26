"""Load precomputed embeddings and provide similarity search."""

import json
from pathlib import Path

import numpy as np


class SemanticLoader:
    _instance = None

    def __init__(self, artifacts_dir: str | Path | None = None):
        self.jd_embedding = None
        self.candidate_embeddings = None
        self.id_index = {}
        
        if artifacts_dir:
            self.load(artifacts_dir)

    @classmethod
    def get_instance(cls) -> "SemanticLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize(cls, artifacts_dir: str | Path) -> None:
        cls._instance = cls(artifacts_dir)

    def load(self, artifacts_dir: str | Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        jd_path = artifacts_dir / "jd_embedding.npy"
        cand_path = artifacts_dir / "candidate_embeddings.npy"
        idx_path = artifacts_dir / "candidate_id_index.json"
        
        if not (jd_path.exists() and cand_path.exists() and idx_path.exists()):
            # Graceful degradation if artifacts not found
            return

        self.jd_embedding = np.load(jd_path)
        self.candidate_embeddings = np.load(cand_path)
        with open(idx_path, "r", encoding="utf-8") as f:
            self.id_index = json.load(f)
            
    def get_similarity(self, candidate_id: str) -> float | None:
        if self.jd_embedding is None or self.candidate_embeddings is None:
            return None
            
        idx = self.id_index.get(candidate_id)
        if idx is None:
            return None
            
        cand_emb = self.candidate_embeddings[idx]
        
        # Since embeddings were normalized during precomputation, dot product == cosine similarity
        sim = float(np.dot(self.jd_embedding, cand_emb))
        
        # Map from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def compute_rich_similarities(self, rich_texts: list[str], jd_text: str) -> list[float]:
        if not rich_texts:
            return []
            
        if not hasattr(self, "model") or self.model is None:
            import os
            # Prevent network requests in sandboxed environment
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
        jd_emb = self.model.encode(jd_text, normalize_embeddings=True)
        cand_embs = self.model.encode(rich_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        
        similarities = np.dot(cand_embs, jd_emb)
        return [float(max(0.0, min(1.0, (sim + 1.0) / 2.0))) for sim in similarities]
