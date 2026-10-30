import json
import re
import numpy as np
from typing import List, Dict, Tuple
from retrieval.faiss_index import DualViewFAISSIndex
from retrieval.trust_scoring import TrustScoreFilter
from llm.prompts import PhishGuardPrompts
from sanitizer.text_sanitizer import TextSanitizer

class PhishGuardRAGEngine:
    """
    Model M6 / PhishGuard-RAG Engine.
    Combines Dual-View FAISS Retriever + Trust Scoring + Sanitizer + LLM Inference.
    Supports Retrain-Free Hot-Patching (Contribution C3).
    """
    def __init__(self, index: DualViewFAISSIndex, llm_classifier=None, top_k: int = 3):
        self.index = index
        self.llm_classifier = llm_classifier
        self.trust_filter = TrustScoreFilter(trust_threshold=0.5)
        self.top_k = top_k

    def predict_single(self, email_text: str) -> Dict:
        # 1. Sanitize text
        canonical_text = TextSanitizer.sanitize(email_text)

        # 2. Dual-view retrieval
        raw_neighbors = self.index.search_dual_view(email_text, top_k=self.top_k)

        # 3. Trust scoring filter
        trusted_neighbors = self.trust_filter.filter_neighbors(raw_neighbors)

        # 4. Fallback ensemble / heuristic RAG decision if local LLM is not instantiated
        if self.llm_classifier is None:
            # Weighted nearest-neighbor classification score
            if not trusted_neighbors:
                pred_label = 1 if any(w in canonical_text.lower() for w in ['urgent', 'password', 'verify', 'click', 'account']) else 0
                confidence = 0.80
            else:
                phish_score = sum(n["weighted_score"] for n in trusted_neighbors if n["label"] == 1)
                legit_score = sum(n["weighted_score"] for n in trusted_neighbors if n["label"] == 0)
                total = phish_score + legit_score + 1e-5
                p_phish = phish_score / total
                pred_label = 1 if p_phish >= 0.5 else 0
                confidence = float(max(p_phish, 1 - p_phish))

            evidence = [n["text_canonical"][:80] for n in trusted_neighbors]
            return {
                "verdict": "Phishing" if pred_label == 1 else "Legitimate",
                "label": pred_label,
                "confidence": round(confidence, 4),
                "evidence_phrases": evidence,
                "retrieved_neighbors": trusted_neighbors
            }

        # 5. Build prompt and run LLM classifier
        prompt = PhishGuardPrompts.build_rag_prompt(canonical_text, trusted_neighbors)
        # Call LLM wrapper (mock or actual generator)
        llm_out = self.llm_classifier(prompt)
        return llm_out

    def predict(self, texts: List[str]) -> np.ndarray:
        results = [self.predict_single(t) for t in texts]
        return np.array([r["label"] for r in results])

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        results = [self.predict_single(t) for t in texts]
        return np.array([r["confidence"] if r["label"] == 1 else (1.0 - r["confidence"]) for r in results])

    def hot_patch_update(self, attack_texts: List[str], attack_labels: List[int]):
        """
        Hot-patching: Appends new attack samples directly to the RAG index without gradient updates.
        """
        self.index.hot_patch_add(attack_texts, attack_labels)
