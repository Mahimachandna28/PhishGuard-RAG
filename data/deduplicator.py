import pandas as pd
import numpy as np
import re
from typing import Tuple

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

class DatasetDeduplicator:
    """
    Exact and near-deduplication module (MinHash LSH / embedding similarity).
    Deduplicates within datasets and across datasets to prevent data leakage.
    """

    @staticmethod
    def exact_deduplicate(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        initial_len = len(df)
        df_clean = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        print(f"[Exact Dedup] Removed {initial_len - len(df_clean)} exact duplicate rows. (Remaining: {len(df_clean)})")
        return df_clean

    @staticmethod
    def get_shingles(text: str, k: int = 3) -> set:
        words = re.findall(r'\w+', text.lower())
        return set(" ".join(words[i:i+k]) for i in range(len(words)-k+1))

    @classmethod
    def minhash_deduplicate(cls, df: pd.DataFrame, text_col: str = "text", threshold: float = 0.85, num_perm: int = 128) -> pd.DataFrame:
        """
        MinHash LSH near-deduplication.
        """
        if not HAS_DATASKETCH:
            print("[MinHash Dedup] datasketch package not installed. Fallback to exact deduplication.")
            return cls.exact_deduplicate(df, text_col=text_col)

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes = {}
        keep_indices = []

        for idx, row in df.iterrows():
            text = row[text_col]
            shingles = cls.get_shingles(text)
            if not shingles:
                keep_indices.append(idx)
                continue

            m = MinHash(num_perm=num_perm)
            for s in shingles:
                m.update(s.encode('utf-8'))

            # Query LSH
            result = lsh.query(m)
            if not result:
                lsh.insert(f"id_{idx}", m)
                keep_indices.append(idx)

        df_dedup = df.iloc[keep_indices].reset_index(drop=True)
        print(f"[MinHash Dedup] Removed {len(df) - len(df_dedup)} near-duplicate samples at similarity threshold >= {threshold}.")
        return df_dedup

    @classmethod
    def cross_deduplicate(cls, df_a: pd.DataFrame, df_b: pd.DataFrame, text_col: str = "text") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Removes overlapping texts between Dataset A and Dataset B.
        """
        texts_a = set(df_a[text_col].values)
        initial_b = len(df_b)
        df_b_clean = df_b[~df_b[text_col].isin(texts_a)].reset_index(drop=True)
        print(f"[Cross Dedup] Removed {initial_b - len(df_b_clean)} cross-dataset duplicate samples from Dataset B.")
        return df_a, df_b_clean
