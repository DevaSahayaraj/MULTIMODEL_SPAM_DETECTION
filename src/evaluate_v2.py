from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from train_models import evaluate


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed_v2"
MODEL_DIR = ROOT / "models_v2"
OUTPUT_DIR = ROOT / "results" / "v2"

SELECTED_MODEL = "linear_svm"
THRESHOLD = 0.5


def main():
    splits = {
        name: pd.read_csv(
            DATA_DIR / f"{name}.csv",
            keep_default_na=False,
        )
        for name in ["train", "validation", "test"]
    }

    # Confirm that identical extracted text does not cross splits.
    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = (
            set(splits[first]["text_hash"])
            & set(splits[second]["text_hash"])
        )
        if overlap:
            raise ValueError(
                f"Found {len(overlap)} overlapping hashes "
                f"between {first} and {second}."
            )

    test = splits["test"]
    labels = pd.to_numeric(
        test["label"], errors="raise"
    ).to_numpy(dtype=int)

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Test data must contain ham and spam.")

    vectorizer = joblib.load(MODEL_DIR / "tfidf.joblib")
    model = joblib.load(
        MODEL_DIR / f"{SELECTED_MODEL}.joblib"
    )

    features = vectorizer.transform(test["text"])
    spam_index = list(model.classes_).index(1)
    scores = model.predict_proba(features)[:, spam_index]

    if not np.isfinite(scores).all():
        raise ValueError("Model produced invalid scores.")

    predictions = (scores >= THRESHOLD).astype(int)
    metrics = evaluate(labels, scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": SELECTED_MODEL,
        "threshold": THRESHOLD,
        "test_emails": len(test),
        **metrics,
    }

    with (OUTPUT_DIR / "selected_model_test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["HAM", "SPAM"],
        digits=4,
        zero_division=0,
    )

    (OUTPUT_DIR / "selected_model_test_report.txt").write_text(
        report, encoding="utf-8"
    )

    details = test.copy()
    details["spam_score"] = scores
    details["prediction"] = predictions
    details["error_type"] = np.select(
        [
            (labels == 0) & (predictions == 1),
            (labels == 1) & (predictions == 0),
        ],
        ["Legitimate marked spam", "Spam missed"],
        default="Correct",
    )

    details.to_csv(
        OUTPUT_DIR / "selected_model_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    details.loc[predictions != labels].to_csv(
        OUTPUT_DIR / "selected_model_test_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nModel: {SELECTED_MODEL}")
    print(f"Test emails: {len(test):,}")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1:        {metrics['f1']:.2%}")
    print(
        "Legitimate marked spam:",
        metrics["ham_marked_spam"],
    )
    print("Spam missed:", metrics["spam_missed"])
    print("\n", report)
    print(f"Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()