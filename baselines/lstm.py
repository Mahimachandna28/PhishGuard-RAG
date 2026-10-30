import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
from evaluation.metrics import evaluate_predictions

class LSTMClassifierNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        out, (hn, cn) = self.lstm(embedded)
        hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        logits = self.fc(hidden)
        return logits

class SimpleVocab:
    def __init__(self, max_tokens: int = 20000):
        self.max_tokens = max_tokens
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts: list):
        counter = Counter()
        for t in texts:
            counter.update(str(t).lower().split())
        most_common = counter.most_common(self.max_tokens - 2)
        for w, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, text: str, max_len: int = 128) -> list:
        tokens = str(text).lower().split()[:max_len]
        ids = [self.word2idx.get(w, 1) for w in tokens]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

class TextSeqDataset(Dataset):
    def __init__(self, texts: list, labels: list, vocab: SimpleVocab, max_len: int = 128):
        self.data = [vocab.encode(t, max_len=max_len) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {"input_ids": torch.tensor(self.data[idx], dtype=torch.long)}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

class M2LSTMClassifier:
    """
    Model M2: PyTorch Bidirectional LSTM Baseline.
    """
    def __init__(self, vocab_size: int = 20000, max_len: int = 128, embed_dim: int = 128, hidden_dim: int = 128):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.vocab = SimpleVocab(max_tokens=vocab_size)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None

    def fit(self, train_texts: list, train_labels: np.ndarray, epochs: int = 3, batch_size: int = 32, lr: float = 1e-3):
        self.vocab.fit(train_texts)
        self.net = LSTMClassifierNet(vocab_size=len(self.vocab.word2idx), embed_dim=self.embed_dim, hidden_dim=self.hidden_dim).to(self.device)
        
        train_ds = TextSeqDataset(train_texts, train_labels, self.vocab, self.max_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(epochs):
            self.net.train()
            total_loss = 0
            for batch in train_loader:
                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                logits = self.net(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[M2 LSTM] Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(train_loader):.4f}")
        print(f"[M2 LSTM] Finished training in {time.time() - start:.2f}s")

    def predict(self, texts: list) -> np.ndarray:
        self.net.eval()
        ds = TextSeqDataset(texts, None, self.vocab, self.max_len)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input_ids"].to(self.device)
                logits = self.net(inputs)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return np.array(preds)

    def predict_proba(self, texts: list) -> np.ndarray:
        self.net.eval()
        ds = TextSeqDataset(texts, None, self.vocab, self.max_len)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        probas = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input_ids"].to(self.device)
                logits = self.net(inputs)
                p = torch.softmax(logits, dim=1)[:, 1]
                probas.extend(p.cpu().numpy())
        return np.array(probas)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({"net_state": self.net.state_dict(), "vocab": self.vocab}, filepath)
        print(f"[M2 LSTM] Saved model weights to {filepath}")

    def load(self, filepath: str):
        data = torch.load(filepath, map_location=self.device)
        self.vocab = data["vocab"]
        self.net = LSTMClassifierNet(vocab_size=len(self.vocab.word2idx), embed_dim=self.embed_dim, hidden_dim=self.hidden_dim).to(self.device)
        self.net.load_state_dict(data["net_state"])
        print(f"[M2 LSTM] Loaded model weights from {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M2: PyTorch Bidirectional LSTM")
    parser.add_argument("--train-data", type=str, default="data/dataset_a.csv", help="Path to training CSV")
    parser.add_argument("--test-data", type=str, default="data/dataset_a.csv", help="Path to test CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--save-path", type=str, default="saved_models/m2_lstm.pt", help="Model save path")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_data)
    df_test = pd.read_csv(args.test_data)

    text_col = "text" if "text" in df_train.columns else "text_combined"
    label_col = "label"

    m2 = M2LSTMClassifier()
    m2.fit(df_train[text_col].tolist(), df_train[label_col].values, epochs=args.epochs)
    m2.save(args.save_path)

    preds = m2.predict(df_test[text_col].tolist())
    probs = m2.predict_proba(df_test[text_col].tolist())
    metrics = evaluate_predictions(df_test[label_col].values, preds, probs)
    print("\n[M2 Test Results]", metrics)
