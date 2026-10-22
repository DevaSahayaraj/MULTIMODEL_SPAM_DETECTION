from itertools import combinations
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "emails.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
SEED = 42


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "ham_marked_spam": int(fp),
        "spam_missed": int(fn),
        "true_ham": int(tn),
        "true_spam": int(tp),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, keep_default_na=False)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain text and label columns.")

    df = df.loc[df["text"].str.strip().ne("")].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 (ham) and 1 (spam).")

    # A shared split makes comparisons between configurations fair.
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED,
    )

    validation_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df["label"],
        random_state=SEED,
    )

    for name, split in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        split.to_csv(
            DATA_PATH.parent / f"{name}.csv", index=False
        )
        print(f"{name}: {len(split):,} emails")

    # Learn vocabulary and IDF weights from training emails only.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=20000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    x_train = vectorizer.fit_transform(train_df["text"])
    x_validation = vectorizer.transform(validation_df["text"])
    x_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label"].to_numpy()

    models = {
        "naive_bayes": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
        # Calibration supplies probabilities for soft voting.
        "linear_svm": CalibratedClassifierCV(
            estimator=LinearSVC(
                max_iter=10000,
                random_state=SEED,
            ),
            method="sigmoid",
            cv=3,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=SEED,
        ),
    }

    probabilities = {"validation": {}, "test": {}}

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(x_train, y_train)

        # Explicitly locate the spam class probability column.
        spam_index = list(model.classes_).index(1)

        probabilities["validation"][name] = model.predict_proba(
            x_validation
        )[:, spam_index]

        probabilities["test"][name] = model.predict_proba(
            x_test
        )[:, spam_index]

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"Saved {name}")

    joblib.dump(vectorizer, MODEL_DIR / "tfidf.joblib")

    configurations = []
    names = list(models)

    for size in range(1, len(names) + 1):
        for members in combinations(names, size):
            configurations.append({
                "id": "__".join(members),
                "models": list(members),
                "type": "individual" if size == 1 else "hybrid",
                "method": "probability_average",
                "threshold": 0.5,
            })

    with (MODEL_DIR / "configurations.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configurations, file, indent=2)

    for split_name, split_df in [
        ("validation", validation_df),
        ("test", test_df),
    ]:
        rows = []

        for configuration in configurations:
            # Equal-weight soft voting for every model subset.
            averaged = np.mean(
                [
                    probabilities[split_name][name]
                    for name in configuration["models"]
                ],
                axis=0,
            )

            rows.append({
                "configuration": configuration["id"],
                "type": configuration["type"],
                "model_count": len(configuration["models"]),
                **evaluate(split_df["label"], averaged),
            })

        results = pd.DataFrame(rows).sort_values(
            ["f1", "precision"],
            ascending=False,
        )
        results.to_csv(
            RESULT_DIR / f"{split_name}_metrics.csv",
            index=False,
        )

        if split_name == "validation":
            print("\nTop configurations on validation data:")
            print(
                results[
                    ["configuration", "precision", "recall", "f1"]
                ].head(10).to_string(index=False)
            )

    print(f"\nComplete: {len(configurations)} configurations.")
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Evaluation results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()