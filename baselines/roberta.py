import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import numpy as np
import time

class RoBERTaDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

class M5RoBERTaClassifier:
    """
    Model M5: Fine-tuned RoBERTa-base Classifier.
    """
    def __init__(self, model_name: str = "roberta-base", max_len: int = 128):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, train_texts: list, train_labels: np.ndarray, val_texts: list = None, val_labels: np.ndarray = None, epochs: int = 2, batch_size: int = 16, lr: float = 1e-5):
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=self.max_len)
        train_ds = RoBERTaDataset(train_encodings, train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        start = time.time()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            print(f"[M5 RoBERTa] Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(train_loader):.4f}")
        print(f"[M5 RoBERTa] Finished training in {time.time() - start:.2f}s")

    def predict(self, texts: list, batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_len)
        ds = RoBERTaDataset(encodings)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        return np.array(preds)

    def predict_proba(self, texts: list, batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_len)
        ds = RoBERTaDataset(encodings)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probas = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                p = torch.softmax(outputs.logits, dim=1)[:, 1]
                probas.extend(p.cpu().numpy())
        return np.array(probas)
