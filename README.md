# PhishGuard-RAG

Cross-dataset and adversarial evaluation of fine-tuned and retrieval-augmented phishing email detectors.

**Type:** research and model-development project. There is no frontend.
**Authors:** [TBD: names, roles, affiliations, e-mails]
**Guide / course:** [TBD]
**Status:** see "Project status" below. Results are not reported here until they come from saved runs.

---

## 1. What this project asks

Phishing detectors are usually trained and tested on a single dataset. A high score then says little about new mail sources or about emails an attacker has edited on purpose. This project compares six detectors under one leakage-controlled protocol and asks:

- **RQ1.** How do classical, recurrent, transformer, and LLM-based detectors compare on clean email from one dataset?
- **RQ2.** How much performance is lost when a model trained on one email dataset is tested on the other, once overlap is removed?
- **RQ3.** Do mutual-information feature fusion, two-view retrieval, or input sanitization improve robustness to adversarial emails?

This is an empirical comparison. It does not claim a new learning algorithm.

## 2. Rules for working on this repository

- This README is the source of truth. Do not change it, or existing working code, without approval.
- Use Antigravity to write and organize code. Use Google Colab (NVIDIA T4 GPU) for heavy training.
- Test sets are locked after splitting and used only for final evaluation.
- Never write numbers into a paper or slide unless they come from a saved result file in `results/`.

## 3. Datasets

| | Dataset A | Dataset B |
|---|---|---|
| Name | Phishing Email Dataset | Phishing and Legitimate Emails Dataset for ML 2026 |
| Owner | naserabdullahalam | kuladeep19 (version 1) |
| Link | https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset | https://www.kaggle.com/datasets/kuladeep19/phishing-and-legitimate-emails-dataset |
| DOI | 10.34740/kaggle/ds/5074342 | not given |
| File used | `phishing_email.csv` (combined) | `phishing_legit_dataset_KD_10000.csv` (2.72 MB) |
| Rows | 82,486 | 10,000 |
| Columns | 2: `text_combined`, `label` | 5: `text`, `label`, `phishing_type`, `severity`, `confidence` |
| Unique texts | 82,078 | 9,956 |
| Cleaning | 1 whitespace-only message and 408 duplicates removed, leaving 82,077 | 44 duplicates; final count [TBD] |
| Labels | original: 39,595 legitimate (0), 42,891 phishing (1); recount after cleaning [TBD] | 1 = phishing, 0 = legitimate; counts [TBD] |

Notes:

- Dataset A's package lists seven files: `CEAS_08.csv`, `Enron.csv`, `Ling.csv`, `Nazario.csv`, `Nigerian_Fraud.csv`, `SpamAssassin.csv`, and `phishing_email.csv`. Do not add their row counts together. For reference, `CEAS_08.csv` has 39,154 rows with columns `body`, `label`, `urls` and no missing values.
- Dataset A has no source column, so per-source scores need extra work (see section 8).
- Dataset B: use only `text` as model input. `phishing_type`, `severity` and `confidence` would leak the label. Dataset card statistics: `phishing_type` is 40% legitimate and 10% credential_harvesting; `severity` is 45% high and 42% low; `confidence` has mean 0.90, SD 0.08, min 0.65, quartiles 0.84 / 0.92 / 0.97, max 1.00. These describe the metadata, not the label counts.
- Dataset B may be script-labelled and is small. State this as a limitation.

### Pilot dataset (not part of the main experiments)

UCI SMS Spam Collection: 5,572 messages (4,825 legitimate, 747 spam), 80/20 split.

| Metric | TF-IDF + LR | DistilBERT | Difference |
|---|---|---|---|
| Accuracy | 98.57% | 99.28% | +0.71 |
| Precision | 96.50% | 98.62% | +2.12 |
| Recall | 92.62% | 95.97% | +3.35 |
| F1 | 94.52% | 97.28% | +2.76 |

This was a single run on a different task. It motivates the design and is not a result of the main study. A MiniLM + FAISS prototype in the same pilot showed that nearest neighbours can include both classes, so retrieval is used as context for a fine-tuned model and not as a majority vote.

## 4. Models

| ID | Model | Hardware |
|---|---|---|
| B0 | Trivial baseline: logistic regression on length, punctuation count, URL count (shortcut check) | CPU |
| M1 | TF-IDF + logistic regression | CPU |
| M2 | LSTM | T4 GPU |
| M3 | DistilBERT | T4 GPU |
| M4 | DistilBERT + mutual-information feature fusion, k = 0 (control), 50, 100, 300, 500 | T4 GPU |
| M5 | RoBERTa-base | T4 GPU |
| M6 | QLoRA fine-tuned Qwen2.5-3B-Instruct + two-view retrieval | T4 GPU |
| M7 | NFDLM: literature reference only, not trained on email | none |

**M4.** MI selects the top-k TF-IDF features for the label. The selected vector is concatenated with the pooled DistilBERT vector. The TF-IDF vectorizer and the MI ranking are fitted on training data only. The best k is chosen on a validation split, not on the test set.

**M6 settings.** Base model `Qwen/Qwen2.5-3B-Instruct`; 4-bit NF4 quantization; LoRA rank 16, alpha 32, dropout 0.05; trained on a Colab T4.

**Retrieval (two views).**

- Semantic view: `all-MiniLM-L6-v2` embeddings in a FAISS index.
- Lexical view: TF-IDF representation in a separate sparse similarity index.
- Fusion: reciprocal rank fusion, RRF(d) = sum over views of 1 / (c + rank), with proposed c = 60.
- Both indexes are built from the training split of the source dataset only.
- M6 can read training emails through retrieval and the other models cannot. Compare RAG on and off.

**Not yet decided:** learning rates, batch sizes, epochs, maximum sequence length, number of retrieved examples n, near-duplicate method and threshold, decoding settings. Record every value in `research/experiment_plan.md` before running.

## 5. Components

- **Sanitizer** (`src/sanitizer.py`): normalize Unicode and whitespace; remove non-printable control characters; keep URLs, e-mail addresses, numbers and suspicious keywords; if a URL or address is replaced by a placeholder, keep the original as separate features; wrap the email in delimiters and tell the LLM it is data, not instructions. Do not strip URLs.
- **Adversarial test set:** built from the phishing emails of each locked test set. The label stays phishing.

| ID | Attack | How it is produced |
|---|---|---|
| A1 | Paraphrase, sentence reordering | [TBD] |
| A2 | Character swaps, typos | [TBD] |
| A3 | URL obfuscation | [TBD] |
| A4 | Urgency and authority wording | [TBD] |
| A5 | LLM-generated phishing | [TBD: use a generator that is not a detector] |
| A6 | Prompt-injection attempts | [TBD: fixed templates] |

Read a random sample of each category to confirm the scam is still present. Never use adversarial emails for training, tuning, or retrieval.

## 6. Evaluation protocol

- Splits: 80/20 train/test per dataset, plus a validation part held out of training [TBD: size, stratified?, seed].
- Runs: A to A, B to B (within), A to B, B to A (cross), and an adversarial copy of each test set.
- Cross-dataset leakage control: remove test emails that appear in the other dataset (exact or near-duplicate) and report how many were removed.
- Metrics: accuracy, precision, recall, F1 (phishing is the positive class), confusion matrix, inference time per email, F1 drop under attack, recall per attack category.
- Statistics: 95% bootstrap intervals; McNemar's test with Holm correction; several seeds where time allows.
- Ablations: M4 over k; M6 retrieval off, semantic only, lexical only, both; sanitizer on and off.
- Error analysis: group mistakes by cause (short emails, marketing mail, truncated long emails, fluent LLM-written phishing).

## 7. Repository structure

```
PhishGuard-RAG/
├── README.md
├── requirements.txt
├── data/
│   ├── dataset_a/
│   ├── dataset_b/
│   └── processed/
├── src/
│   ├── preprocessing.py
│   ├── sanitizer.py
│   ├── dataset_splitting.py
│   ├── feature_selection.py
│   ├── retrieval.py
│   ├── training.py
│   └── evaluation.py
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_tfidf_logistic_regression.ipynb
│   ├── 03_lstm.ipynb
│   ├── 04_distilbert.ipynb
│   ├── 05_distilbert_mutual_information.ipynb
│   ├── 06_roberta.ipynb
│   ├── 07_qlora_rag.ipynb
│   └── 08_cross_dataset_robustness.ipynb
├── models/
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   └── robustness/
└── research/
    ├── methodology.md
    └── experiment_plan.md
```

Adjust this to the repository you already have. Do not rename or replace working files without checking them first.

## 8. Known limitations (state them in the paper)

1. Both datasets are public and may not resemble mail in a real organization.
2. Dataset A merges sources and has no source column, so models may use source style. B0 and a leave-one-source-out run (if the per-source files can be matched) only partly address this.
3. Dataset B is small and may have script-made labels. A third human-labelled dataset would strengthen the cross-dataset claim.
4. Adversarial emails are made by us and are not adapted to any model.
5. Long emails are truncated to the model input length.
6. M6 is a 3B model trained on one T4 GPU.
7. F1 is only a proxy for protection, so recall is reported separately.

## 9. Project status

| Item | Status |
|---|---|
| Pilot on SMS (LR vs DistilBERT) | done, values in section 3 |
| MiniLM + FAISS retrieval prototype | done as a prototype |
| Main runs on both email datasets | run once, **no results saved**: rerun and save to `results/` |
| Adversarial tests | performed, details and numbers not saved |
| Research paper (Springer one-column, then IEEE two-column) | draft written, results are slots |
| Presentation | 10 slides, no result numbers |

## 10. How to reproduce

1. Download both datasets from the links above into `data/dataset_a/` and `data/dataset_b/`.
2. Run `src/preprocessing.py` and `src/dataset_splitting.py` with a fixed seed. Save the split files.
3. Run the baselines first (B0, M1, M3), then M2, M5, M4, and M6 on Colab (T4).
4. Run the cross-dataset and adversarial notebooks.
5. Save every metric table to `results/metrics/` as CSV. Quote only these files in the paper.
6. Record library versions (scikit-learn, Transformers, PEFT, bitsandbytes, FAISS, sentence-transformers) in `requirements.txt`.

## 11. References

DistilBERT (Sanh et al., 2019); RoBERTa (Liu et al., 2019); LoRA (Hu et al., 2021); QLoRA (Dettmers et al., 2023); RAG (Lewis et al., 2020); FAISS (Johnson et al.); Sentence-BERT (Reimers and Gurevych, 2019); reciprocal rank fusion (Cormack et al., 2009); Qwen2.5 technical report (2024); NFDLM (Saurabh et al., 2022); Li et al. (2025, urlBERT); Said et al. (2024); Kim et al. (2025). Full entries are in the paper. Verify every reference before submission.

## 12. License and citation

License: [TBD]. Citation: [TBD after publication].
