import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
import time
from features.mi_selector import MutualInformationSelector

class DistilBERT_MI_Net(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", mi_dim: int = 300, num_classes: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        combined_dim = hidden_dim + mi_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, mi_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        fused = torch.cat((cls_embedding, mi_features), dim=1)
        logits = self.classifier(fused)
        return logits

class DistilBertMIDataset(Dataset):
    def __init__(self, encodings, mi_features, labels=None):
        self.encodings = encodings
        self.mi_features = torch.tensor(mi_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "mi_features": self.mi_features[idx]
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

class M4DistilBertMIClassifier:
    """
    Model M4: DistilBERT + Mutual Information Feature Fusion (Contribution C4).
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", top_k: int = 300, max_len: int = 128):
        self.model_name = model_name
        self.top_k = top_k
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mi_selector = MutualInformationSelector(top_k=top_k)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None

    def fit(self, train_texts: list, train_labels: np.ndarray, val_texts: list = None, val_labels: np.ndarray = None, epochs: int = 2, batch_size: int = 16, lr: float = 2e-5):
        train_mi_feat = self.mi_selector.fit_transform(train_texts, train_labels)
        self.net = DistilBERT_MI_Net(model_name=self.model_name, mi_dim=self.top_k, num_classes=2).to(self.device)
        
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_len)
        train_ds = DistilBertMIDataset(train_encodings, train_mi_feat, train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.net.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(epochs):
            self.net.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                mi_feats = batch["mi_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.net(input_ids, attention_mask, mi_feats)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            print(f"[M4 DistilBERT+MI] Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(train_loader):.4f}")
        print(f"[M4 DistilBERT+MI] Finished training in {time.time() - start:.2f}s")

    def predict(self, texts: list, batch_size: int = 32) -> np.ndarray:
        self.net.eval()
        mi_feats = self.mi_selector.transform(texts)
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_len)
        ds = DistilBertMIDataset(encodings, mi_feats)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                mf = batch["mi_features"].to(self.device)
                logits = self.net(input_ids, attention_mask, mf)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return np.array(preds)

    def predict_proba(self, texts: list, batch_size: int = 32) -> np.ndarray:
        self.net.eval()
        mi_feats = self.mi_selector.transform(texts)
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_len)
        ds = DistilBertMIDataset(encodings, mi_feats)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probas = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                mf = batch["mi_features"].to(self.device)
                logits = self.net(input_ids, attention_mask, mf)
                p = torch.softmax(logits, dim=1)[:, 1]
                probas.extend(p.cpu().numpy())
        return np.array(probas)
