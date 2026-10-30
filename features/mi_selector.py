import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from typing import List, Tuple
from features.structural import StructuralFeatureExtractor

class MutualInformationSelector:
    """
    Mutual Information feature selector (Contribution C4).
    Computes I(feature; label) strictly on the training set to prevent data leakage.
    Extracts top-k lexical (TF-IDF) + structural features for DistilBERT + MI model (M4).
    """

    def __init__(self, top_k: int = 300, max_tfidf_features: int = 2000):
        self.top_k = top_k
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english')
        self.selected_indices = []
        self.feature_names = []

    def fit(self, train_texts: List[str], train_labels: np.ndarray):
        """
        Fit TF-IDF and select top-k features using Mutual Information on training split ONLY.
        """
        # 1. TF-IDF features
        X_tfidf = self.tfidf_vectorizer.fit_transform(train_texts).toarray()
        tfidf_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]

        # 2. Structural features
        df_struct = StructuralFeatureExtractor.extract_features_df(train_texts)
        X_struct = df_struct.values
        struct_names = df_struct.columns.tolist()

        # Combine candidate feature matrix
        X_combined = np.hstack([X_tfidf, X_struct])
        all_names = tfidf_names + struct_names

        # Compute Mutual Information score
        mi_scores = mutual_info_classif(X_combined, train_labels, discrete_features='auto', random_state=42)

        # Select top-k features
        top_k_indices = np.argsort(mi_scores)[::-1][:self.top_k]
        self.selected_indices = top_k_indices
        self.feature_names = [all_names[i] for i in top_k_indices]

        print(f"[MI Selector] Selected top {len(self.feature_names)} features based on Mutual Information.")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts into top-k MI feature vectors.
        """
        X_tfidf = self.tfidf_vectorizer.transform(texts).toarray()
        df_struct = StructuralFeatureExtractor.extract_features_df(texts)
        X_struct = df_struct.values
        X_combined = np.hstack([X_tfidf, X_struct])
        return X_combined[:, self.selected_indices]

    def fit_transform(self, train_texts: List[str], train_labels: np.ndarray) -> np.ndarray:
        self.fit(train_texts, train_labels)
        return self.transform(train_texts)
