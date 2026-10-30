import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any

URGENCY_WORDS = {
    'immediately', 'urgent', 'suspended', 'verify', 'account', 'security',
    'unauthorized', 'password', 'login', 'click', 'bank', 'action', 'required',
    'winner', 'prize', 'congratulations', 'claimed', 'expire', 'limited', 'risk'
}

class StructuralFeatureExtractor:
    """
    Extracts structural, lexical, and HTML features from raw email text for MI feature selection (M4).
    """

    @staticmethod
    def extract_features_single(text: str) -> Dict[str, float]:
        if not isinstance(text, str):
            text = ""
        
        text_lower = text.lower()
        words = text_lower.split()
        num_words = max(len(words), 1)

        # 1. URL metrics
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)
        url_count = len(urls)
        has_ip_url = int(any(re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) for url in urls))

        # 2. HTML to text ratio & tag count
        html_tags = re.findall(r'<[^>]+>', text)
        html_tag_count = len(html_tags)
        html_to_text_ratio = len("".join(html_tags)) / max(len(text), 1)

        # 3. Urgency word count & ratio
        urgency_count = sum(1 for w in words if w in URGENCY_WORDS)
        urgency_ratio = urgency_count / num_words

        # 4. Link text mismatch (simple heuristic: http in anchor text vs domain)
        mismatch_count = 0
        anchors = re.findall(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', text, re.IGNORECASE)
        for href, anchor_text in anchors:
            if 'http' in anchor_text.lower() and href.lower() not in anchor_text.lower():
                mismatch_count += 1

        # 5. Character structural ratios
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_char_count / max(len(text), 1)
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / max(len(text), 1)

        return {
            "url_count": float(url_count),
            "has_ip_url": float(has_ip_url),
            "html_tag_count": float(html_tag_count),
            "html_to_text_ratio": float(html_to_text_ratio),
            "urgency_count": float(urgency_count),
            "urgency_ratio": float(urgency_ratio),
            "mismatch_count": float(mismatch_count),
            "special_char_ratio": float(special_char_ratio),
            "uppercase_ratio": float(uppercase_ratio),
            "text_length": float(len(text))
        }

    @classmethod
    def extract_features_df(cls, texts: List[str]) -> pd.DataFrame:
        features = [cls.extract_features_single(t) for t in texts]
        return pd.DataFrame(features)
