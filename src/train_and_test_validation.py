# ==============================================================================
# PhishGuard-RAG: Train, Validation and Test Pipeline (TF-IDF + Logistic Regression)
# ==============================================================================

import os
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------------------------
# 1. Dataset Loading (Supports Local Paths & Google Colab Uploads)
# ------------------------------------------------------------------------------
def locate_file(filename, fallback_paths):
    if os.path.exists(filename):
        return filename
    for p in fallback_paths:
        if os.path.exists(p):
            return p
    return None

path_a = locate_file(
    "phishing_email.csv",
    ["data/dataset_a/phishing_email.csv", "../data/dataset_a/phishing_email.csv"]
)

path_b = locate_file(
    "phishing_legit_dataset_KD_10000.csv",
    ["data/dataset_b/phishing_legit_dataset_KD_10000.csv", "../data/dataset_b/phishing_legit_dataset_KD_10000.csv"]
)

if not path_a or not path_b:
    try:
        from google.colab import files
        print("Dataset files not found in standard directories. Please upload:")
        uploaded = files.upload()
        path_a = "phishing_email.csv"
        path_b = "phishing_legit_dataset_KD_10000.csv"
    except ImportError:
        raise FileNotFoundError(
            f"Could not locate dataset files.\n"
            f"Expected 'phishing_email.csv' (or in data/dataset_a/) and "
            f"'phishing_legit_dataset_KD_10000.csv' (or in data/dataset_b/)."
        )

print(f"Loading Dataset A from: {path_a}")
df_a = pd.read_csv(path_a)

print(f"Loading Dataset B from: {path_b}")
df_b = pd.read_csv(path_b)

print("Dataset A Shape:", df_a.shape)
print("Dataset B Shape:", df_b.shape)

# ------------------------------------------------------------------------------
# 2. Clean the datasets
# ------------------------------------------------------------------------------
def clean_dataset(df, text_column, label_column):
    # Keep only the required columns
    df = df[[text_column, label_column]].copy()

    # Rename columns for consistency
    df.columns = ["text", "label"]

    # Remove missing values
    df = df.dropna(subset=["text", "label"])

    # Convert email text into string
    df["text"] = df["text"].astype(str)

    # Normalize extra spaces and new lines
    df["text"] = df["text"].apply(
        lambda x: re.sub(r"\s+", " ", x).strip()
    )

    # Remove empty messages
    df = df[df["text"] != ""]

    # Check for duplicate text with conflicting labels
    conflicting = df.groupby("text")["label"].nunique()
    conflicting_texts = conflicting[conflicting > 1].index
    print("Conflicting duplicate messages:", len(conflicting_texts))

    # Remove conflicting messages to avoid ambiguous labels
    df = df[~df["text"].isin(conflicting_texts)]

    # Remove exact duplicate messages
    df = df.drop_duplicates(subset=["text"])

    # Ensure labels are binary integers
    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)

# Clean Dataset A
data_a = clean_dataset(
    df_a,
    text_column="text_combined" if "text_combined" in df_a.columns else "text",
    label_column="label"
)

# Clean Dataset B
data_b = clean_dataset(
    df_b,
    text_column="text",
    label_column="label"
)

print("\nCleaned Dataset A:", data_a.shape)
print("Cleaned Dataset B:", data_b.shape)

# ------------------------------------------------------------------------------
# 3. Display class distribution
# ------------------------------------------------------------------------------
print("\nDataset A Class Distribution:")
print(data_a["label"].value_counts())

print("\nDataset B Class Distribution:")
print(data_b["label"].value_counts())

# ------------------------------------------------------------------------------
# 4. Split into train, validation and test
# ------------------------------------------------------------------------------
def split_dataset(df):
    X = df["text"]
    y = df["label"]

    # First separate 20% for final testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Separate 10% of total data for validation (0.125 * 0.80 = 0.10)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.125,
        random_state=42,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Split Dataset A
A_train, A_val, A_test, Ay_train, Ay_val, Ay_test = split_dataset(data_a)

# Split Dataset B
B_train, B_val, B_test, By_train, By_val, By_test = split_dataset(data_b)

# ------------------------------------------------------------------------------
# 5. Display split sizes
# ------------------------------------------------------------------------------
print("\nDATASET A SPLIT")
print("Training  :", len(A_train))
print("Validation:", len(A_val))
print("Testing   :", len(A_test))

print("\nDATASET B SPLIT")
print("Training  :", len(B_train))
print("Validation:", len(B_val))
print("Testing   :", len(B_test))

print("\nPreprocessing and splitting completed!")

# ------------------------------------------------------------------------------
# 6. Function to train and evaluate a model
# ------------------------------------------------------------------------------
def train_and_evaluate(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    dataset_name
):
    print("\n" + "=" * 55)
    print("RESULTS FOR", dataset_name)
    print("=" * 55)

    # 1. Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words=None,
        min_df=2,
        sublinear_tf=True
    )

    # Fit only on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform validation data
    X_val_tfidf = vectorizer.transform(X_val)

    # 2. Hyperparameter tuning
    C_values = [0.1, 0.5, 1, 2, 5, 10]

    best_model = None
    best_C = None
    best_f1 = -1

    print("\nVALIDATION RESULTS")

    for c in C_values:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        # Train using training data
        model.fit(X_train_tfidf, y_train)

        # Predict validation data
        val_predictions = model.predict(X_val_tfidf)

        # Calculate validation F1-score
        val_f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        print(f"C = {c:<4} | Validation F1 = {val_f1:.4f}")

        # Save the best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = c
            best_model = model

    print("\nBest C:", best_C)
    print("Best Validation F1:", round(best_f1, 4))

    # 3. Refit using training + validation
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])

    # Create a fresh vectorizer
    final_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words=None,
        min_df=2,
        sublinear_tf=True
    )

    X_train_final_tfidf = final_vectorizer.fit_transform(X_train_final)

    # Train final model with the selected C
    final_model = LogisticRegression(
        C=best_C,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    final_model.fit(X_train_final_tfidf, y_train_final)

    # 4. Final testing
    X_test_tfidf = final_vectorizer.transform(X_test)

    start_time = time.time()
    test_predictions = final_model.predict(X_test_tfidf)
    prediction_time = time.time() - start_time

    # 5. Calculate evaluation metrics
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)

    print("\nFINAL TEST RESULTS")
    print("------------------------------")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("Prediction Time:", round(prediction_time, 4), "seconds")

    # Detailed classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Legitimate", "Phishing"],
            zero_division=0
        )
    )

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(dataset_name + " - Confusion Matrix")
    
    os.makedirs("results/confusion_matrices", exist_ok=True)
    cm_path = f"results/confusion_matrices/{dataset_name.lower().replace(' ', '_')}_cm.png"
    plt.savefig(cm_path, bbox_inches="tight")
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

    # 7. Save trained model and vectorizer
    os.makedirs("saved_models", exist_ok=True)
    model_filename = f"saved_models/{dataset_name.lower().replace(' ', '_')}"
    joblib.dump(final_model, model_filename + "_model.pkl")
    joblib.dump(final_vectorizer, model_filename + "_vectorizer.pkl")
    print(f"Saved model and vectorizer to {model_filename}_*.pkl")

    return {
        "Dataset": dataset_name,
        "Best C": best_C,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Prediction Time (seconds)": prediction_time
    }

# ------------------------------------------------------------------------------
# 7. Run Pipeline for Dataset A and Dataset B
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    results_a = train_and_evaluate(
        A_train, A_val, A_test,
        Ay_train, Ay_val, Ay_test,
        "Dataset A"
    )

    results_b = train_and_evaluate(
        B_train, B_val, B_test,
        By_train, By_val, By_test,
        "Dataset B"
    )

    results_df = pd.DataFrame([results_a, results_b])
    print("\nFINAL COMPARISON")
    print(results_df.to_string(index=False))

    os.makedirs("results/metrics", exist_ok=True)
    results_df.to_csv("results/metrics/phishguard_results.csv", index=False)
    print("\nSaved comparison results to results/metrics/phishguard_results.csv")
    print("All training, validation, and testing completed!")
