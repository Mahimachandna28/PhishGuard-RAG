import random
from typing import List, Tuple
from retrieval.faiss_index import DualViewFAISSIndex

class IndexPoisoningAttacks:
    """
    RAG Index Poisoning attack simulator (Section 10).
    Inserts a controlled ratio (1%, 5%, 10%) of corrupted entries into the RAG vector index.
    """

    @classmethod
    def poison_index(cls, index: DualViewFAISSIndex, phishing_texts: List[str], poison_ratio: float = 0.05):
        num_poison = int(len(phishing_texts) * poison_ratio)
        if num_poison == 0:
            return

        poison_samples = random.sample(phishing_texts, num_poison)
        # Flip label of phishing samples to 0 (Legitimate) to confuse retriever
        poison_labels = [0] * num_poison
        # Set low trust score if trust filter is active
        trust_scores = [0.2] * num_poison

        index.hot_patch_add(poison_samples, poison_labels, trust_scores=trust_scores)
        print(f"[Index Poisoning] Injected {num_poison} ({poison_ratio*100:.1f}%) poisoned samples into RAG index.")
