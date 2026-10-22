from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "validation.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "error_analysis"


def spam_scores(model, features):
    spam_index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, spam_index]


def error_types(actual, predicted):
    return np.select(
        [
            (actual == 0) & (predicted == 1),
            (actual == 1) & (predicted == 0),
        ],
        [
            "Legitimate marked as spam",
            "Spam missed",
        ],
        default="Correct",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    df["label"] = pd.to_numeric(
        df["label"], errors="raise"
    ).astype(int)

    vectorizer = joblib.load(MODEL_DIR / "tfidf.joblib")
    svm = joblib.load(MODEL_DIR / "linear_svm.joblib")
    xgboost = joblib.load(MODEL_DIR / "xgboost.joblib")

    features = vectorizer.transform(df["text"])

    svm_scores = spam_scores(svm, features)
    xgb_scores = spam_scores(xgboost, features)
    hybrid_scores = (svm_scores + xgb_scores) / 2

    actual = df["label"].to_numpy()
    svm_predictions = (svm_scores >= 0.5).astype(int)
    hybrid_predictions = (hybrid_scores >= 0.5).astype(int)

    report = df.copy()
    report["actual_label"] = np.where(actual == 1, "SPAM", "HAM")

    report["svm_spam_score"] = svm_scores
    report["xgboost_spam_score"] = xgb_scores
    report["hybrid_spam_score"] = hybrid_scores

    report["svm_prediction"] = np.where(
        svm_predictions == 1, "SPAM", "HAM"
    )
    report["hybrid_prediction"] = np.where(
        hybrid_predictions == 1, "SPAM", "HAM"
    )

    report["svm_error"] = error_types(actual, svm_predictions)
    report["hybrid_error"] = error_types(
        actual, hybrid_predictions
    )

    svm_wrong = svm_predictions != actual
    hybrid_wrong = hybrid_predictions != actual

    report["comparison"] = np.select(
        [
            svm_wrong & hybrid_wrong,
            svm_wrong & ~hybrid_wrong,
            ~svm_wrong & hybrid_wrong,
        ],
        [
            "Both wrong",
            "Hybrid fixes SVM error",
            "Hybrid introduces error",
        ],
        default="Both correct",
    )

    exports = {
        "all_validation_predictions.csv": report,
        "errors_either_model.csv": report.loc[
            svm_wrong | hybrid_wrong
        ],
        "svm_errors.csv": report.loc[svm_wrong],
        "hybrid_errors.csv": report.loc[hybrid_wrong],
        "model_disagreements.csv": report.loc[
            svm_predictions != hybrid_predictions
        ],
    }

    for filename, frame in exports.items():
        frame.to_csv(
            OUTPUT_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )

    summary = []

    for name, predictions in [
        ("Linear SVM", svm_predictions),
        ("Linear SVM + XGBoost", hybrid_predictions),
    ]:
        false_positives = int(
            ((actual == 0) & (predictions == 1)).sum()
        )
        false_negatives = int(
            ((actual == 1) & (predictions == 0)).sum()
        )

        summary.append({
            "model": name,
            "legitimate_marked_spam": false_positives,
            "spam_missed": false_negatives,
            "total_errors": false_positives + false_negatives,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        OUTPUT_DIR / "summary.csv", index=False
    )

    print("\nValidation error summary:")
    print(summary_df.to_string(index=False))

    print("\nComparison:")
    print(report["comparison"].value_counts().to_string())

    print(f"\nReports saved to: {OUTPUT_DIR}")
    print(
        "Review errors_either_model.csv to inspect "
        "the emails and their prediction scores."
    )


if __name__ == "__main__":
    main()