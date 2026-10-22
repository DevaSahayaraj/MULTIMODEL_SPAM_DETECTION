from pathlib import Path

import pandas as pd

from prepare_dataset import parse_email


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "parsing_audit"


def main():
    summary = pd.read_csv(
        AUDIT_DIR / "email_summary.csv",
        keep_default_na=False,
        dtype={"audit_id": str},
    )

    rows = []

    for _, row in summary.iterrows():
        audit_id = row["audit_id"].zfill(3)
        original = AUDIT_DIR / audit_id / "original.eml"

        subject, body, text = parse_email(original.read_bytes())

        rows.append({
            "audit_id": audit_id,
            "email_id": row["email_id"],
            "subject": subject,
            "old_body_characters": row["old_body_characters"],
            "new_body_characters": len(body),
            "old_replacement_characters": (
                row["old_replacement_characters"]
            ),
            "new_replacement_characters": body.count("\ufffd"),
            "new_body": body,
        })

    report = pd.DataFrame(rows)
    output = AUDIT_DIR / "parser_comparison.csv"
    report.to_csv(output, index=False, encoding="utf-8-sig")

    print(report[[
        "audit_id",
        "old_body_characters",
        "new_body_characters",
        "new_replacement_characters",
    ]].to_string(index=False))

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()