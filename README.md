# Multi-Model Spam Email Detection

A Streamlit application for classifying email as **spam** or **ham (legitimate)** using five machine learning classifiers and every unique combination of those classifiers.

**5 trained classifiers · 26 soft-voting ensembles · 31 selectable configurations**

The selected V2 Linear SVM achieved **98.87% accuracy** and **98.16% spam F1** on the project's 888-email test split. These results describe this corpus and split, not guaranteed performance on new email.

## Features

- Analyze an email by entering its subject and body.
- Upload an `.eml` file using the parser snapshot saved during V2 training.
- Upload a UTF-8 CSV for batch prediction, up to 500 emails.
- Select an individual classifier or a hybrid combination.
- Compare all 31 configurations for the same input.
- Download predictions as CSV.
- Report empty input and unrecognized vocabulary as unavailable.
- Inspect evaluation metrics, confusion matrices, and misclassified emails through the included scripts.

## Models and ensembles

| Classifier | Implementation |
|---|---|
| Multinomial Naive Bayes | `MultinomialNB` |
| Logistic Regression | `LogisticRegression` |
| Linear SVM | `LinearSVC` with sigmoid probability calibration |
| Random Forest | `RandomForestClassifier` |
| XGBoost | `XGBClassifier` |

Email text is represented with word-level TF-IDF features using unigrams and bigrams. The vocabulary is fitted on training data only.

Hybrids use **equal-weight soft voting**: the system averages the spam probabilities returned by the selected classifiers. An average score of at least **0.5** is classified as spam.

| Number of classifiers in a configuration | Configurations |
|---|---:|
| 1 | 5 |
| 2 | 10 |
| 3 | 10 |
| 4 | 5 |
| 5 | 1 |
| **Total** | **31** |

The count is `2^5 - 1 = 31`. The 26 hybrid configurations reuse the five fitted classifiers; they are not separately trained models. Different voting methods or weights are outside this count.

## Dataset

The project uses the [Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/).

Download these archives and place them, without extracting them, in `data/raw/`:

| Archive | Contents |
|---|---|
| [20030228_easy_ham.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2) | Legitimate emails |
| [20030228_easy_ham_2.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2) | Additional legitimate emails |
| [20030228_hard_ham.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2) | Legitimate emails resembling spam |
| [20030228_spam.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2) | Spam emails |
| [20050311_spam_2.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2) | Additional spam emails |

Labels are **0 = ham** and **1 = spam**. Only email text is used as model input; labels, archive names, file identifiers, and duplicate hashes are excluded from the features.

The source corpus is historical. Consult its accompanying documentation and terms before reuse or redistribution. Dataset archives and extracted email text should not be committed to this repository.

## Project layout

```text
multimodel_spam_detection/
├── app.py
├── test_system.py
├── requirements.txt
├── README.md
├── src/
│   ├── prepare_dataset.py
│   ├── train_models.py
│   ├── analyze_errors.py
│   ├── audit_email_parsing.py
│   ├── check_parser_update.py
│   ├── retrain_v2.py
│   └── evaluate_v2.py
├── data/                        # Generated/downloaded locally
│   ├── raw/
│   ├── processed/
│   └── processed_v2/
├── models/                      # Initial fitted artifacts
├── models_v2/                   # Artifacts loaded by the application
└── results/                     # Evaluation and audit outputs
```

## Installation

Use **Python 3.11**. A GPU is not required.

From the project root, run the following in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the virtual environment's Python directly avoids requiring PowerShell activation.

The main dependencies are NumPy, pandas, SciPy, scikit-learn, XGBoost, joblib, and Streamlit. Install the repository's recorded versions for the closest match to the development environment.

## Build the model artifacts

After downloading the dataset archives, run:

```powershell
.\.venv\Scripts\python.exe src\prepare_dataset.py
.\.venv\Scripts\python.exe src\train_models.py
.\.venv\Scripts\python.exe src\retrain_v2.py
```

The first script parses email and removes duplicate or conflicting text. Initial training creates stratified **70% training, 15% validation, and 15% test** splits, trains the five classifiers, and evaluates the 31 configurations.

The V2 script reprocesses emails with the current parser while preserving the original split assignments of retained emails. It checks for newly overlapping text, conflicting labels, and duplicates, logs excluded rows, and retrains into `models_v2/`. It compares baseline and V2 on the same surviving validation emails.

> The reported baseline-to-V2 experiment used the original plain-text-preferred parser for baseline training and the revised parser for V2. If `prepare_dataset.py` already contains the revised parser, a fresh run uses that parser from the start and will not recreate the historical baseline comparison. Exact historical reproduction requires the original parser version and matching dependency versions.

The application requires these generated artifacts:

```text
models_v2/
├── tfidf.joblib
├── naive_bayes.joblib
├── logistic_regression.joblib
├── linear_svm.joblib
├── random_forest.joblib
├── xgboost.joblib
├── configurations.json
└── prepare_dataset_snapshot.py
```

Only load model artifacts and parser snapshots from a trusted source: joblib files and Python modules can execute code when loaded.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL printed in the terminal. The application loads `models_v2/` and defaults to **Linear SVM**.

### Pasted email

Enter the subject and body, select a configuration, and click **Analyze email**. Enable **Compare all 31 configurations** to see every configuration's prediction.

### Email file

Upload an `.eml` file smaller than 10 MB. The V2 parser extracts subject and plain-text/HTML content, excluding attachments and attached email messages. HTML is parsed as text; the application does not need to visit email links.

### Batch CSV

Upload a UTF-8 CSV smaller than 10 MB with 1–500 rows and either:

- A `text` column containing the complete email text; an optional `subject` column is used only for display.
- Both `subject` and `body` columns, which are combined for classification.

Example:

```csv
subject,body
Project meeting,Please join the project review tomorrow.
Claim your prize,Click now to claim your cash reward.
```

Use the application's downloadable template to get started. Prediction scores range from 0 to 1; they are estimates rather than guarantees. Empty emails and inputs with no recognized vocabulary receive `UNAVAILABLE`.

## Evaluation

Run the selected V2 model's test evaluation:

```powershell
.\.venv\Scripts\python.exe src\evaluate_v2.py
```

Outputs are saved under `results/v2/`, including aggregate metrics, a classification report, predictions, and errors.

`test_system.py` checks the initial `models/` artifacts, initial splits, and all 31 configurations. It does **not** evaluate V2 unless its paths are updated accordingly.

### Selected V2 Linear SVM: test results

| Metric | Result |
|---|---:|
| Test emails | 888 |
| Accuracy | **98.87%** |
| Spam precision | **97.80%** |
| Spam recall | **98.52%** |
| Spam F1 | **98.16%** |
| Correct classifications | 878 |
| Legitimate emails marked spam | 6 |
| Spam emails missed | 4 |

Confusion matrix:

| Actual class | Predicted ham | Predicted spam |
|---|---:|---:|
| Ham | 611 | 6 |
| Spam | 4 | 267 |

### Validation comparison

Both versions were compared on 887 validation emails. The parser update improved F1 for **30 of 31 configurations**.

| Configuration | Baseline F1 | V2 F1 | V2 false positives | V2 missed spam |
|---|---:|---:|---:|---:|
| Linear SVM | 96.68% | **97.43%** | 8 | 6 |
| Linear SVM + XGBoost | 96.82% | **97.41%** | 6 | 8 |
| All five classifiers | 96.44% | **97.03%** | 6 | 10 |

Linear SVM was selected using validation performance. Its F1 advantage over SVM + XGBoost is very small and does not establish a general performance advantage.

The test split had previously been evaluated during baseline development, so these results should be treated as an internal held-out comparison. A fresh external dataset is needed for a stronger final assessment. Avoid repeatedly using test results to guide tuning.

## Parser improvements

Error analysis identified multipart emails where plain-text parts contained only a footer or a message asking the reader to view HTML. The revised parser collects usable plain-text and HTML body content and removes identical alternatives.

This recovered missing content in six of the 22 audited emails. In particular, two missed spam emails gained their actual promotional bodies instead of only server footers.

Available audit commands:

```powershell
.\.venv\Scripts\python.exe src\analyze_errors.py
.\.venv\Scripts\python.exe src\audit_email_parsing.py
.\.venv\Scripts\python.exe src\check_parser_update.py
```

These scripts target the baseline error-analysis workflow. Generated reports may contain full email text; keep them out of the public repository.

## Limitations

- The historical corpus may not represent current spam campaigns or modern email traffic.
- Some messages have unresolved character-decoding problems, especially when charset declarations are missing or incorrect.
- Plain-text and HTML alternatives may overlap without being exact duplicates.
- Image-only content is not read using OCR; attachments are excluded.
- The system does not inspect sender reputation, authentication records, attachment contents, or live URL reputation.
- Text alone cannot reliably establish whether a promotional message was solicited.
- Deduplication checks identical normalized text, not all near-duplicate messages or campaign overlap.
- Spam scores are not guaranteed to be equally calibrated across classifiers; averaging them is the current ensemble method.
- The application is a local project prototype, not a production email gateway or a guarantee of phishing protection.

## Before publishing to GitHub

Include source code, `requirements.txt`, this README, and aggregate metrics you choose to publish. Add screenshots using synthetic sample emails if desired.

Exclude `.venv/`, downloaded archives, processed email datasets, model artifacts, uploaded emails, parser audit folders, and row-level prediction/error reports. Record working dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

Do not commit secrets or credentials. Choose a code license and add its corresponding `LICENSE` file before claiming that the repository uses that license. Dataset rights are separate from the project's code license.

## Author

**Deva Sahayaraj** — [GitHub](https://github.com/DevaSahayaraj)
