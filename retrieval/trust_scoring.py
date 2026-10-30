import numpy as np
from typing import List, Dict

class TrustScoreFilter:
    """
    Trust-Scored Retrieval (Contribution C5).
    Filters or down-weights outlier or low-provenance vector index entries to defend against index poisoning.
    """

    def __init__(self, trust_threshold: float = 0.5):
        self.trust_threshold = trust_threshold

    def filter_neighbors(self, neighbors: List[Dict]) -> List[Dict]:
        """
        Down-weights low-trust neighbors and filters out entries below trust_threshold.
        """
        trusted_neighbors = []
        for n in neighbors:
            trust = n.get("trust_score", 1.0)
            if trust >= self.trust_threshold:
                # Modulate similarity score by provenance trust factor
                n["weighted_score"] = n["score"] * trust
                trusted_neighbors.append(n)

        # Re-sort by weighted score
        trusted_neighbors.sort(key=lambda x: x.get("weighted_score", x["score"]), reverse=True)
        return trusted_neighbors
