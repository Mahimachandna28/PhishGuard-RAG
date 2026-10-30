# ==============================================================================
# PhishGuard-RAG: Pilot Phishing & Spam Detection Benchmark
# Models: TF-IDF + Logistic Regression, MiniLM + FAISS RAG, DistilBERT
# ==============================================================================

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
import faiss

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ------------------------------------------------------------------------------
# 1. Environment & Setup Check
# ------------------------------------------------------------------------------
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------------------------
# 2. Dataset Download & Loading (SMS Spam Pilot)
# ------------------------------------------------------------------------------
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

urllib.request.urlretrieve(url, "sms.zip")
with zipfile.ZipFile("sms.zip", "r") as z:
    z.extractall(data_dir)

print("Data folder contents:", os.listdir(data_dir))

df = pd.read_csv(
    os.path.join(data_dir, "SMSSpamCollection"),
    sep="\t",
    header=None,
    names=["label", "text"]
)

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\n--- Dataset Sample ---")
print(df.head())
print("\n--- Label Counts ---")
print(df["label"].value_counts())

# Train/Test Split (80/20 Stratified)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTraining samples: {len(train_df)}")
print(f"Testing samples : {len(test_df)}")

# ------------------------------------------------------------------------------
# 3. Model 1: Baseline TF-IDF + Logistic Regression
# ------------------------------------------------------------------------------
print("\n==================================================")
print("Running Baseline: TF-IDF + Logistic Regression")
print("==================================================")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]

print("Train matrix shape:", X_train.shape)
print("Test matrix shape :", X_test.shape)

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
lr.fit(X_train, y_train)

baseline_pred = lr.predict(X_test)

print("\n--- Baseline Classification Report ---")
print(classification_report(
    y_test,
    baseline_pred,
    target_names=["Legitimate", "Spam/Phishing"]
))

baseline_results = {
    "Accuracy": accuracy_score(y_test, baseline_pred),
    "Precision": precision_score(y_test, baseline_pred),
    "Recall": recall_score(y_test, baseline_pred),
    "F1": f1_score(y_test, baseline_pred)
}
print("Baseline Results:", baseline_results)

# ------------------------------------------------------------------------------
# 4. Retrieval-Augmented Generation (RAG) Prototype with FAISS
# ------------------------------------------------------------------------------
print("\n==================================================")
print("Building RAG Prototype (all-MiniLM-L6-v2 + FAISS)")
print("==================================================")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

knowledge_texts = train_df["text"].tolist()
knowledge_labels = train_df["label"].tolist()

knowledge_embeddings = embedding_model.encode(
    knowledge_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)
knowledge_embeddings = np.array(knowledge_embeddings).astype("float32")
print("Embedding shape:", knowledge_embeddings.shape)

dimension = knowledge_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(knowledge_embeddings)
print("Knowledge base size:", index.ntotal)

def retrieve(query, k=5):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    scores, indices = index.search(query_embedding, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text": knowledge_texts[idx],
            "label": knowledge_labels[idx],
            "similarity": float(score)
        })
    return results

def rag_predict(query, k=5):
    results = retrieve(query, k)
    spam_score = 0
    legit_score = 0
    for r in results:
        weight = max(r["similarity"], 0)
        if r["label"] == 1:
            spam_score += weight
        else:
            legit_score += weight
    total = spam_score + legit_score
    spam_probability = spam_score / total if total > 0 else 0
    prediction = int(spam_probability >= 0.5)
    return prediction, spam_probability, results

sample_query = "URGENT! Your bank account has been suspended. Click this link to verify."
pred, prob, retrieved_res = rag_predict(sample_query)
print("\nSample Query:", sample_query)
print("Prediction:", "SPAM/PHISHING" if pred else "LEGITIMATE", f"(Spam probability: {prob:.3f})")

test_messages = [
    "URGENT! Your bank account has been suspended. Click this link to verify.",
    "Hey bro, are we still meeting at 5?",
    "Congratulations! You have won a free prize. Call now to claim.",
    "Please send me the notes from today's class.",
    "Your account will be closed unless you verify your information immediately."
]

print("\n--- Testing RAG Classifier on Samples ---")
for message in test_messages:
    prediction, probability, _ = rag_predict(message)
    print(f"\nMessage: {message}")
    print("Prediction:", "SPAM/PHISHING" if prediction else "LEGITIMATE")
    print(f"Probability: {probability:.3f}")

# ------------------------------------------------------------------------------
# 5. Model 3: DistilBERT Fine-Tuning
# ------------------------------------------------------------------------------
print("\n==================================================")
print("Fine-Tuning DistilBERT (distilbert-base-uncased)")
print("==================================================")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

train_hf = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
test_hf = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_hf = train_hf.map(tokenize, batched=True)
test_hf = test_hf.map(tokenize, batched=True)

train_hf = train_hf.remove_columns(["text"])
test_hf = test_hf.remove_columns(["text"])
train_hf.set_format("torch")
test_hf.set_format("torch")

training_args = TrainingArguments(
    output_dir="./phishing_distilbert",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=test_hf
)

trainer.train()

# Evaluate DistilBERT
predictions = trainer.predict(test_hf)
pred_labels = np.argmax(predictions.predictions, axis=1)

print("\n--- DistilBERT Classification Report ---")
print(classification_report(y_test, pred_labels, target_names=["Legitimate", "Spam/Phishing"]))

distilbert_results = {
    "Accuracy": accuracy_score(y_test, pred_labels),
    "Precision": precision_score(y_test, pred_labels),
    "Recall": recall_score(y_test, pred_labels),
    "F1": f1_score(y_test, pred_labels)
}
print("DistilBERT Results:", distilbert_results)
