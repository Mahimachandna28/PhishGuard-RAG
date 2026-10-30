import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

class FuzzyLayer(nn.Module):
    """
    Fuzzy Membership Layer for Neuro-Fuzzy Deep Learning Model (NFDLM).
    Computes Gaussian membership values over feature dimension.
    """
    def __init__(self, in_features: int, num_rules: int = 16):
        super().__init__()
        self.in_features = in_features
        self.num_rules = num_rules
        self.centers = nn.Parameter(torch.randn(num_rules, in_features))
        self.sigmas = nn.Parameter(torch.ones(num_rules, in_features))

    def forward(self, x):
        # x shape: (batch_size, in_features)
        # expand dims to compute membership per rule
        x_exp = x.unsqueeze(1) # (batch_size, 1, in_features)
        c_exp = self.centers.unsqueeze(0) # (1, num_rules, in_features)
        s_exp = torch.clamp(self.sigmas.unsqueeze(0), min=1e-4) # (1, num_rules, in_features)

        fuzzy_membership = torch.exp(-torch.pow((x_exp - c_exp) / s_exp, 2)) # (batch_size, num_rules, in_features)
        rule_firing = torch.mean(fuzzy_membership, dim=2) # (batch_size, num_rules)
        return rule_firing

class NFDLM_Net(nn.Module):
    """
    Neuro-Fuzzy Deep Learning Model (NFDLM) Architecture.
    Combines Transformer feature representations with a Fuzzy Neural Logic layer.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", num_rules: int = 16, num_classes: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.fuzzy_layer = FuzzyLayer(in_features=hidden_dim, num_rules=num_rules)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + num_rules, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_dim)
        fuzzy_firing = self.fuzzy_layer(cls_rep) # (batch_size, num_rules)
        combined = torch.cat((cls_rep, fuzzy_firing), dim=1)
        logits = self.fc(combined)
        return logits

class NFDLMDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx])
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

class M7NFDLMClassifier:
    """
    Model M7: Neuro-Fuzzy Deep Learning Model (Reference Paper Reproduction).
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", num_rules: int = 16, max_len: int = 128):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.net = NFDLM_Net(model_name=model_name, num_rules=num_rules)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, train_texts: list, train_labels: np.ndarray, val_texts: list = None, val_labels: np.ndarray = None, epochs: int = 2, batch_size: int = 16, lr: float = 2e-5):
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_len)
        train_ds = NFDLMDataset(train_encodings, train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(epochs):
            self.net.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.net(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[M7 NFDLM] Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(train_loader):.4f}")
        print(f"[M7 NFDLM] Finished training in {time.time() - start:.2f}s")

    def predict(self, texts: list, batch_size: int = 32) -> np.ndarray:
        self.net.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_len)
        ds = NFDLMDataset(encodings)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.net(input_ids, attention_mask)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return np.array(preds)

    def predict_proba(self, texts: list, batch_size: int = 32) -> np.ndarray:
        self.net.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_len)
        ds = NFDLMDataset(encodings)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probas = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.net(input_ids, attention_mask)
                p = torch.softmax(logits, dim=1)[:, 1]
                probas.extend(p.cpu().numpy())
        return np.array(probas)
