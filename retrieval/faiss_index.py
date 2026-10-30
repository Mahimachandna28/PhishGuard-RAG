import faiss
import numpy as np
from typing import List, Dict, Tuple
from retrieval.embeddings import EmbeddingExtractor
from sanitizer.text_sanitizer import TextSanitizer

class DualViewFAISSIndex:
    """
    Sanitize-then-dual-view FAISS Retriever & Hot-Patchable Index (Contributions C2 & C3).
    Maintains separate vector indices for Raw view and Canonical view of training emails.
    """
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.extractor = EmbeddingExtractor(model_name=embedding_model)
        self.raw_index = None
        self.canonical_index = None
        self.metadata = [] # List of dicts {"text_raw", "text_canonical", "label", "id", "trust_score"}

    def build_index(self, train_texts: List[str], train_labels: List[int]):
        """
        Build dual-view indices strictly using training data.
        """
        self.metadata = []
        raw_texts = []
        canonical_texts = []

        for idx, text in enumerate(train_texts):
            raw, canonical = TextSanitizer.dual_view(text)
            raw_texts.append(raw)
            canonical_texts.append(canonical)
            self.metadata.append({
                "id": idx,
                "text_raw": raw,
                "text_canonical": canonical,
                "label": int(train_labels[idx]),
                "trust_score": 1.0
            })

        print(f"[FAISS Dual-View] Encoding {len(train_texts)} samples for Raw and Canonical indices...")
        raw_embeds = self.extractor.encode(raw_texts)
        canonical_embeds = self.extractor.encode(canonical_texts)

        dim = raw_embeds.shape[1]
        self.raw_index = faiss.IndexFlatIP(dim) # Inner product for normalized cosine sim
        self.canonical_index = faiss.IndexFlatIP(dim)

        self.raw_index.add(raw_embeds)
        self.canonical_index.add(canonical_embeds)

        print(f"[FAISS Dual-View] Successfully built dual-view index with {self.raw_index.ntotal} entries.")

    def search_dual_view(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves top-k nearest neighbor labeled emails using dual-view (raw + canonical).
        Returns deduplicated merged top matches.
        """
        query_raw, query_canonical = TextSanitizer.dual_view(query_text)
        
        q_raw_embed = self.extractor.encode([query_raw])
        q_canon_embed = self.extractor.encode([query_canonical])

        # Search raw index
        scores_r, indices_r = self.raw_index.search(q_raw_embed, top_k)
        # Search canonical index
        scores_c, indices_c = self.canonical_index.search(q_canon_embed, top_k)

        results_dict = {}
        
        for sim, idx in zip(scores_r[0], indices_r[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(sim)
                item["view"] = "raw"
                results_dict[idx] = item

        for sim, idx in zip(scores_c[0], indices_c[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                if idx not in results_dict or sim > results_dict[idx]["score"]:
                    item["score"] = float(sim)
                    item["view"] = "canonical"
                    results_dict[idx] = item

        sorted_neighbors = sorted(results_dict.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return sorted_neighbors

    def hot_patch_add(self, new_texts: List[str], new_labels: List[int], trust_scores: List[float] = None):
        """
        Hot-patching (Contribution C3): Instantly add new attack samples to the index WITHOUT retraining!
        """
        raw_texts = []
        canonical_texts = []

        start_id = len(self.metadata)
        for idx, text in enumerate(new_texts):
            raw, canonical = TextSanitizer.dual_view(text)
            raw_texts.append(raw)
            canonical_texts.append(canonical)
            score = trust_scores[idx] if trust_scores else 1.0
            self.metadata.append({
                "id": start_id + idx,
                "text_raw": raw,
                "text_canonical": canonical,
                "label": int(new_labels[idx]),
                "trust_score": float(score)
            })

        raw_embeds = self.extractor.encode(raw_texts)
        canonical_embeds = self.extractor.encode(canonical_texts)

        self.raw_index.add(raw_embeds)
        self.canonical_index.add(canonical_embeds)

        print(f"[FAISS Hot-Patch] Hot-patched {len(new_texts)} new samples into RAG index! Total index size: {self.raw_index.ntotal}")
