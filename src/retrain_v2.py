from pathlib import Path
import hashlib
import json
import shutil
import tarfile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from prepare_dataset import parse_email
from train_models import evaluate


ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = ROOT / "data" / "processed"
NEW_DATA = ROOT / "data" / "processed_v2"
OLD_MODELS = ROOT / "models"
NEW_MODELS = ROOT / "models_v2"
RESULTS = ROOT / "results" / "v2"


def predict_scores(model, features):
    index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, index]


def main():
    # Check baseline artifacts before starting.
    with (OLD_MODELS / "configurations.json").open(
        encoding="utf-8"
    ) as file:
        configurations = json.load(file)

    names = sorted({
        name
        for configuration in configurations
        for name in configuration["models"]
    })

    if len(names) != 5 or len(configurations) != 31:
        raise ValueError("Expected 5 models and 31 configurations.")

    baseline_vectorizer = joblib.load(OLD_MODELS / "tfidf.joblib")
    baseline_models = {
        name: joblib.load(OLD_MODELS / f"{name}.joblib")
        for name in names
    }

    frames = []

    for split in ["train", "validation", "test"]:
        frame = pd.read_csv(
            OLD_DATA / f"{split}.csv",
            keep_default_na=False,
        )
        frame["split"] = split
        frame["baseline_text"] = frame["text"]
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["label"] = pd.to_numeric(
        data["label"], errors="raise"
    ).astype(int)

    if data.duplicated(["source", "email_id"]).any():
        raise ValueError("Repeated email identity in baseline splits.")

    print("Rebuilding email text...", flush=True)

    for source, group in data.groupby("source"):
        archive_path = ROOT / "data" / "raw" / source

        with tarfile.open(archive_path, "r:bz2") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile()
            }

            for index, row in group.iterrows():
                member = members.get(row["email_id"])

                if member is None:
                    raise ValueError(
                        f"Missing email: {source}/{row['email_id']}"
                    )

                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable email: {row['email_id']}")

                with stream:
                    subject, body, text = parse_email(stream.read())

                data.at[index, "subject"] = subject
                data.at[index, "body"] = body
                data.at[index, "text"] = text
                data.at[index, "text_hash"] = hashlib.sha256(
                    text.casefold().encode("utf-8")
                ).hexdigest()

        print(f"Parsed {source}", flush=True)

    # Remove all members of newly overlapping/conflicting groups.
    # Surviving emails keep their original split assignment.
    grouped = data.groupby("text_hash")
    cross_split_hashes = grouped["split"].nunique()
    cross_split_hashes = cross_split_hashes[
        cross_split_hashes > 1
    ].index

    conflicting_hashes = grouped["label"].nunique()
    conflicting_hashes = conflicting_hashes[
        conflicting_hashes > 1
    ].index

    reasons = pd.Series("", index=data.index)

    def flag(mask, reason):
        reasons.loc[mask] = reasons.loc[mask] + reason + "; "

    flag(data["text"].str.strip().eq(""), "empty text")
    flag(
        data["text_hash"].isin(cross_split_hashes),
        "duplicate across splits",
    )
    flag(
        data["text_hash"].isin(conflicting_hashes),
        "conflicting labels",
    )

    candidates = data.loc[reasons.eq("")]
    duplicate_indices = candidates.index[
        candidates.duplicated(["split", "text_hash"])
    ]
    flag(data.index.isin(duplicate_indices), "duplicate within split")

    removed = data.loc[reasons.ne("")].copy()
    removed["removal_reason"] = reasons.loc[removed.index]

    clean = data.loc[reasons.eq("")].copy()

    for split in ["train", "validation", "test"]:
        subset = clean.loc[clean["split"] == split]
        if set(subset["label"].unique()) != {0, 1}:
            raise ValueError(f"{split} must contain both classes.")

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        first_hashes = set(
            clean.loc[clean["split"] == first, "text_hash"]
        )
        second_hashes = set(
            clean.loc[clean["split"] == second, "text_hash"]
        )
        if first_hashes & second_hashes:
            raise ValueError("Cross-split duplicate check failed.")

    for directory in [NEW_DATA, NEW_MODELS, RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)

    removed.to_csv(RESULTS / "removed_rows.csv", index=False)

    splits = {}
    for split in ["train", "validation", "test"]:
        splits[split] = clean.loc[clean["split"] == split].copy()
        splits[split].drop(
            columns=["baseline_text", "split"]
        ).to_csv(NEW_DATA / f"{split}.csv", index=False)
        print(f"{split}: {len(splits[split]):,} emails")

    print(f"Excluded rows: {len(removed):,}")

    train = splits["train"]
    validation = splits["validation"]

    # Clone settings, not fitted vocabulary or learned model weights.
    vectorizer = clone(baseline_vectorizer)
    x_train = vectorizer.fit_transform(train["text"])
    x_validation = vectorizer.transform(validation["text"])

    # Compare baseline and V2 on the SAME surviving validation emails.
    baseline_features = baseline_vectorizer.transform(
        validation["baseline_text"]
    )

    baseline_scores = {}
    new_scores = {}

    for name in names:
        print(f"Training V2 {name}...", flush=True)

        baseline_scores[name] = predict_scores(
            baseline_models[name], baseline_features
        )

        model = clone(baseline_models[name])
        model.fit(x_train, train["label"].to_numpy())

        new_scores[name] = predict_scores(model, x_validation)
        joblib.dump(model, NEW_MODELS / f"{name}.joblib")

    joblib.dump(vectorizer, NEW_MODELS / "tfidf.joblib")

    with (NEW_MODELS / "configurations.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configurations, file, indent=2)

    # Keep the exact parser implementation used for this experiment.
    shutil.copy2(
        Path(__file__).with_name("prepare_dataset.py"),
        NEW_MODELS / "prepare_dataset_snapshot.py",
    )

    reports = {}

    for version, scores in [
        ("baseline", baseline_scores),
        ("v2", new_scores),
    ]:
        rows = []

        for configuration in configurations:
            if configuration.get("threshold", 0.5) != 0.5:
                raise ValueError(
                    "This comparison expects the original 0.5 threshold."
                )

            averaged = np.mean([
                scores[name] for name in configuration["models"]
            ], axis=0)

            rows.append({
                "configuration": configuration["id"],
                "type": configuration["type"],
                **evaluate(validation["label"], averaged),
            })

        reports[version] = pd.DataFrame(rows)
        reports[version].to_csv(
            RESULTS / f"{version}_validation_metrics.csv",
            index=False,
        )

    comparison = reports["baseline"].merge(
        reports["v2"],
        on=["configuration", "type"],
        suffixes=("_baseline", "_v2"),
        validate="one_to_one",
    )

    comparison["f1_change_percentage_points"] = 100 * (
        comparison["f1_v2"] - comparison["f1_baseline"]
    )
    comparison["false_positive_change"] = (
        comparison["ham_marked_spam_v2"]
        - comparison["ham_marked_spam_baseline"]
    )
    comparison["missed_spam_change"] = (
        comparison["spam_missed_v2"]
        - comparison["spam_missed_baseline"]
    )

    comparison = comparison.sort_values("f1_v2", ascending=False)
    comparison.to_csv(
        RESULTS / "baseline_vs_v2.csv", index=False
    )

    print("\nTop V2 validation configurations:")
    print(comparison[[
        "configuration",
        "f1_baseline",
        "f1_v2",
        "f1_change_percentage_points",
        "ham_marked_spam_v2",
        "spam_missed_v2",
    ]].head(10).to_string(index=False))

    print("\nComplete.")
    print(f"V2 models: {NEW_MODELS}")
    print(f"Comparison: {RESULTS / 'baseline_vs_v2.csv'}")
    print("The application still uses your baseline models.")
    print("Test metrics were not evaluated in this experiment.")


if __name__ == "__main__":
    main()