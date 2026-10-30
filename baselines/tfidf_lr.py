import os
import argparse
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from evaluation.metrics import evaluate_predictions

class M1TfidfLogisticRegression:
    """
    Model M1: TF-IDF + Logistic Regression Baseline.
    """
    def __init__(self, max_features: int = 10000, C: float = 1.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)

    def fit(self, train_texts: list, train_labels: np.ndarray):
        start = time.time()
        X_train = self.vectorizer.fit_transform(train_texts)
        self.model.fit(X_train, train_labels)
        print(f"[M1 TF-IDF+LR] Training finished in {time.time() - start:.2f}s")

    def predict(self, texts: list) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: list) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, filepath)
        print(f"[M1 TF-IDF+LR] Saved trained model to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        print(f"[M1 TF-IDF+LR] Loaded model from {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M1: TF-IDF + Logistic Regression")
    parser.add_argument("--train-data", type=str, default="data/dataset_a.csv", help="Path to training CSV")
    parser.add_argument("--test-data", type=str, default="data/dataset_a.csv", help="Path to test CSV")
    parser.add_argument("--save-path", type=str, default="saved_models/m1_tfidf_lr.joblib", help="Model save path")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_data)
    df_test = pd.read_csv(args.test_data)

    text_col = "text" if "text" in df_train.columns else "text_combined"
    label_col = "label"

    m1 = M1TfidfLogisticRegression()
    m1.fit(df_train[text_col].tolist(), df_train[label_col].values)
    m1.save(args.save_path)

    preds = m1.predict(df_test[text_col].tolist())
    probs = m1.predict_proba(df_test[text_col].tolist())
    metrics = evaluate_predictions(df_test[label_col].values, preds, probs)
    print("\n[M1 Test Results]", metrics)
