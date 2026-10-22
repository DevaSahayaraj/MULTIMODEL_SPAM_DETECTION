from pathlib import Path
from email import policy
from email.parser import BytesParser
import json
import tarfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ERROR_PATH = ROOT / "results" / "error_analysis" / "errors_either_model.csv"
OUTPUT_DIR = ROOT / "results" / "parsing_audit"


def decode_payload(part):
    """Try strict decoding before accepting replacement characters."""
    payload = part.get_payload(decode=True)

    if payload is None:
        content = part.get_payload()
        return (
            content if isinstance(content, str) else "",
            "already_text",
        )

    declared = part.get_content_charset()

    # Broader Chinese decoding is a useful fallback for GB2312 mail.
    candidates = [declared]
    if declared and declared.lower().replace("-", "") in {
        "gb2312", "gbk"
    }:
        candidates.append("gb18030")

    candidates.extend(["utf-8", "ascii"])

    tried = set()

    for encoding in candidates:
        if not encoding or encoding.lower() in tried:
            continue

        tried.add(encoding.lower())

        try:
            return payload.decode(encoding, errors="strict"), encoding
        except (LookupError, UnicodeError):
            continue

    # Preserve evidence of undecodable bytes instead of guessing a language.
    try:
        return (
            payload.decode(declared or "utf-8", errors="replace"),
            f"{declared or 'utf-8'}: replacement fallback",
        )
    except LookupError:
        return (
            payload.decode("utf-8", errors="replace"),
            "utf-8: replacement fallback",
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = pd.read_csv(ERROR_PATH, keep_default_na=False)
    required = {"source", "email_id", "body", "subject"}
    if not required.issubset(errors.columns):
        raise ValueError(f"Missing columns: {required - set(errors.columns)}")

    part_rows = []
    email_rows = []

    for source, group in errors.groupby("source"):
        archive_path = RAW_DIR / source

        if not archive_path.exists():
            raise FileNotFoundError(archive_path)

        with tarfile.open(archive_path, "r:bz2") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile()
            }

            for _, row in group.iterrows():
                email_id = row["email_id"]
                member = members.get(email_id)

                if member is None:
                    raise ValueError(
                        f"Email not found in {source}: {email_id}"
                    )

                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read {email_id}")

                with stream:
                    raw = stream.read()

                message = BytesParser(
                    policy=policy.default
                ).parsebytes(raw)

                key = f"{len(email_rows) + 1:03d}"
                email_dir = OUTPUT_DIR / key
                email_dir.mkdir(exist_ok=True)

                # Local evidence for checking MIME and charset declarations.
                (email_dir / "original.eml").write_bytes(raw)

                text_parts = 0
                attachment_parts = 0

                for index, part in enumerate(message.walk()):
                    if part.get_content_disposition() == "attachment":
                        attachment_parts += 1

                    if part.is_multipart():
                        continue

                    content_type = part.get_content_type()
                    if content_type not in {"text/plain", "text/html"}:
                        continue

                    decoded, used_encoding = decode_payload(part)
                    text_parts += 1

                    # Save HTML as text, so the audit does not render it.
                    preview_name = f"part_{index:02d}.txt"
                    (email_dir / preview_name).write_text(
                        decoded, encoding="utf-8"
                    )

                    part_rows.append({
                        "audit_id": key,
                        "email_id": email_id,
                        "part_index": index,
                        "content_type": content_type,
                        "disposition": part.get_content_disposition(),
                        "declared_charset": part.get_content_charset(),
                        "used_encoding": used_encoding,
                        "transfer_encoding": str(
                            part.get("Content-Transfer-Encoding", "")
                        ),
                        "decoded_characters": len(decoded),
                        "replacement_characters": decoded.count("\ufffd"),
                        "defects": json.dumps(
                            [str(defect) for defect in part.defects]
                        ),
                        "preview_file": f"{key}/{preview_name}",
                    })

                email_rows.append({
                    "audit_id": key,
                    "source": source,
                    "email_id": email_id,
                    "subject": row["subject"],
                    "old_body_characters": len(row["body"]),
                    "old_replacement_characters": row["body"].count("\ufffd"),
                    "text_parts": text_parts,
                    "attachment_parts": attachment_parts,
                    "message_defects": json.dumps(
                        [str(defect) for defect in message.defects]
                    ),
                })

    pd.DataFrame(email_rows).to_csv(
        OUTPUT_DIR / "email_summary.csv", index=False
    )
    pd.DataFrame(part_rows).to_csv(
        OUTPUT_DIR / "mime_parts.csv", index=False
    )

    print(f"Audited {len(email_rows)} emails.")
    print(f"Inspected {len(part_rows)} text/HTML parts.")
    print(f"Reports saved to: {OUTPUT_DIR}")
    print("No datasets or trained models were changed.")


if __name__ == "__main__":
    main()