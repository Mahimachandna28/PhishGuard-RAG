import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class SplitterAndLeakageChecker:
    """
    Stratified train/val/test splitter and strict data leakage verifier.
    Ensures splitting is done AFTER deduplication and verifies zero overlap.
    """

    @staticmethod
    def split_dataset(df: pd.DataFrame, train_size: float = 0.80, val_size: float = 0.10, test_size: float = 0.10, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Splits must sum to 1.0"
        
        X = df["text"]
        y = df["label"]

        # First split: train vs temp (val + test)
        temp_ratio = val_size + test_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_ratio, random_state=seed, stratify=y
        )

        # Second split: val vs test
        val_relative_ratio = val_size / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_relative_ratio), random_state=seed, stratify=y_temp
        )

        train_df = pd.DataFrame({"text": X_train, "label": y_train}).reset_index(drop=True)
        val_df = pd.DataFrame({"text": X_val, "label": y_val}).reset_index(drop=True)
        test_df = pd.DataFrame({"text": X_test, "label": y_test}).reset_index(drop=True)

        print(f"[Data Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        return train_df, val_df, test_df

    @staticmethod
    def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        train_texts = set(train_df["text"].values)
        val_texts = set(val_df["text"].values)
        test_texts = set(test_df["text"].values)

        tv_overlap = len(train_texts.intersection(val_texts))
        tt_overlap = len(train_texts.intersection(test_texts))
        vt_overlap = len(val_texts.intersection(test_texts))

        leakage_detected = (tv_overlap > 0) or (tt_overlap > 0) or (vt_overlap > 0)

        if leakage_detected:
            raise ValueError(f"[DATA LEAKAGE DETECTED!] Train-Val Overlap: {tv_overlap}, Train-Test Overlap: {tt_overlap}, Val-Test Overlap: {vt_overlap}")
        else:
            print("[Leakage Check PASSED] Zero text overlap across Train, Val, and Test splits.")
        return True
