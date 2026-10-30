import os
import pandas as pd
import numpy as np
from typing import Tuple
from sanitizer.text_sanitizer import TextSanitizer

class DatasetLoader:
    """
    Unified dataset loader and preprocessor for Dataset A & Dataset B.
    Handles schema mapping, cleaning, and sanitization.
    """

    @staticmethod
    def load_csv(file_path: str, text_col: str = "text_combined", label_col: str = "label") -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
        df = pd.read_csv(file_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise KeyError(f"Expected columns '{text_col}' and '{label_col}' in {file_path}. Found: {df.columns.tolist()}")

        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        
        # Remove empty texts
        df = df[df["text"] != ""]
        return df.reset_index(drop=True)

    @classmethod
    def load_dataset_a(cls, file_path: str) -> pd.DataFrame:
        return cls.load_csv(file_path, text_col="text_combined", label_col="label")

    @classmethod
    def load_dataset_b(cls, file_path: str) -> pd.DataFrame:
        # Accepts custom text/label column names or standard text/label
        df = pd.read_csv(file_path)
        text_col = "text" if "text" in df.columns else ("text_combined" if "text_combined" in df.columns else df.columns[0])
        label_col = "label" if "label" in df.columns else df.columns[1]
        return cls.load_csv(file_path, text_col=text_col, label_col=label_col)
