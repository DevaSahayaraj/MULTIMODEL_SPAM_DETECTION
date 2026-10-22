from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results"


def main():
    with (MODEL_DIR / "configurations.json").open(
        encoding="utf-8"
    ) as file:
        configurations = json.load(file)

    # Check that every nonempty subset appears exactly once.
    names = {
        name
        for configuration in configurations
        for name in configuration["models"]
    }
    groups = [
        frozenset(configuration["models"])
        for configuration in configurations
    ]

    assert len(names) == 5, "Expected five individual models."
    assert len(configurations) == 31, "Expected 31 configurations."
    assert len(set(groups)) == 31, "Duplicate configurations found."
    assert all(groups), "Empty configuration found."
    assert all(
        len(group) == len(configuration["models"])
        for group, configuration in zip(groups, configurations)
    ), "Repeated model inside a configuration."

    splits = {
        name: pd.read_csv(
            DATA_DIR / f"{name}.csv",
            keep_default_na=False,
        )
        for name in ["train", "validation", "test"]
    }

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = (
            set(splits[first]["text_hash"])
            & set(splits[second]["text_hash"])
        )
        assert not overlap, (
            f"Duplicate text overlaps {first} and {second}."
        )

    test_df = splits["test"]
    assert len(test_df) > 0, "Test dataset is empty."
    assert set(test_df["label"].unique()) == {0, 1}

    vectorizer = joblib.load(MODEL_DIR / "tfidf.joblib")
    features = vectorizer.transform(test_df["text"])
    labels = test_df["label"].to_numpy()

    probabilities = {}

    for name in sorted(names):
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        spam_index = list(model.classes_).index(1)

        scores = model.predict_proba(features)[:, spam_index]

        assert scores.shape == (len(test_df),)
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()

        probabilities[name] = scores
        print(f"PASS: {name}")

    rows = []

    for configuration in configurations:
        scores = np.mean([
            probabilities[name]
            for name in configuration["models"]
        ], axis=0)

        predictions = (
            scores >= configuration.get("threshold", 0.5)
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "configuration": configuration["id"],
            "type": configuration["type"],
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(
                labels, predictions, zero_division=0
            ),
            "recall": recall_score(
                labels, predictions, zero_division=0
            ),
            "f1": f1_score(
                labels, predictions, zero_division=0
            ),
            "ham_marked_spam": int(fp),
            "spam_missed": int(fn),
            "true_ham": int(tn),
            "true_spam": int(tp),
        })

    results = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "system_test_metrics.csv"
    results.to_csv(output, index=False)

    print(
        f"\nPASS: all 31 configurations evaluated "
        f"on {len(test_df):,} test emails."
    )
    print("PASS: no identical text overlaps the saved splits.")
    print("\nIndividual-model test results:")
    print(
        results.loc[
            results["type"] == "individual",
            ["configuration", "accuracy", "precision", "recall", "f1"],
        ].to_string(index=False)
    )
    print(f"\nFull report: {output}")
    print(
        "\nFunctional checks passing does not establish "
        "acceptable detection quality; review the metrics."
    )


if __name__ == "__main__":
    main()