from pathlib import Path
from email import policy
from email.parser import BytesParser
from html import unescape
from html.parser import HTMLParser
import hashlib
import re
import tarfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "emails.csv"


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        elif tag in {"br", "p", "div", "li", "tr"}:
            self.text.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)
        self.text.append(" ")

    def handle_data(self, data):
        if not self.hidden_depth:
            self.text.append(data)


def clean_text(text):
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(html):
    parser = HTMLTextExtractor()
    parser.feed(html)
    return clean_text(" ".join(parser.text))


def decode_part(part):
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except (LookupError, UnicodeError, ValueError):
        pass

    payload = part.get_payload(decode=True)
    if isinstance(payload, bytes):
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")

    return ""


# def parse_email(raw_bytes):
#     message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
#     subject = clean_text(str(message.get("Subject", "")))

#     # Prefer plain text; use HTML when plain text is unavailable.
#     body_part = message.get_body(preferencelist=("plain", "html"))
#     body = ""

#     if body_part is not None:
#         body = decode_part(body_part)
#         if body_part.get_content_type() == "text/html":
#             body = html_to_text(body)

#     body = clean_text(body)
#     text = clean_text(f"{subject} {body}")
#     return subject, body, text


def parse_email(raw_bytes):
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = clean_text(str(message.get("Subject", "")))

    body_blocks = []
    seen = set()

    def collect(part):
        # Exclude attachments and attached/forwarded email messages.
        if part.get_content_disposition() == "attachment":
            return

        if part.get_content_type() == "message/rfc822":
            return

        if part.is_multipart():
            for child in part.iter_parts():
                collect(child)
            return

        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            return

        content = decode_part(part)

        if content_type == "text/html":
            content = html_to_text(content)
        else:
            content = clean_text(content)

        if not content:
            return

        # Remove identical alternatives while keeping distinct content.
        key = content.casefold()
        if key not in seen:
            seen.add(key)
            body_blocks.append(content)

    collect(message)

    body = clean_text(" ".join(body_blocks))
    text = clean_text(f"{subject} {body}")
    return subject, body, text

def archive_label(path):
    name = path.name.lower()

    if "ham" in name:
        return 0
    if "spam" in name:
        return 1

    raise ValueError(f"Cannot determine label for archive: {path.name}")


def main():
    archives = sorted(RAW_DIR.glob("*.tar.bz2"))
    if not archives:
        raise FileNotFoundError(
            f"No .tar.bz2 archives found in {RAW_DIR}. "
            "Place the downloaded archives there without extracting them."
        )

    rows = []
    skipped = 0

    for archive_path in archives:
        label = archive_label(archive_path)
        archive_count = 0

        with tarfile.open(archive_path, mode="r:bz2") as archive:
            for member in archive:
                if not member.isfile():
                    continue

                filename = Path(member.name).name
                if filename.lower().startswith(("cmds", "readme", ".")):
                    continue

                try:
                    email_file = archive.extractfile(member)
                    if email_file is None:
                        skipped += 1
                        continue

                    with email_file:
                        subject, body, text = parse_email(email_file.read())

                    if not text:
                        skipped += 1
                        continue

                    # Case-insensitive, whitespace-normalized duplicate key.
                    fingerprint = hashlib.sha256(
                        text.casefold().encode("utf-8")
                    ).hexdigest()

                    rows.append({
                        "subject": subject,
                        "body": body,
                        "text": text,
                        "label": label,
                        "source": archive_path.name,
                        "email_id": member.name,
                        "text_hash": fingerprint,
                    })
                    archive_count += 1

                except Exception as exc:
                    skipped += 1
                    print(f"Skipped {member.name}: {exc}")

        print(f"{archive_path.name}: read {archive_count:,} emails")

    if not rows:
        raise ValueError("No usable emails were found.")

    df = pd.DataFrame(rows)
    initial_count = len(df)

    # Exclude identical text assigned to both spam and ham.
    label_counts = df.groupby("text_hash")["label"].nunique()
    conflicting_hashes = label_counts[label_counts > 1].index
    conflict_mask = df["text_hash"].isin(conflicting_hashes)
    conflicting_rows = int(conflict_mask.sum())
    df = df.loc[~conflict_mask].copy()

    before_dedup = len(df)
    df = df.drop_duplicates(subset="text_hash").reset_index(drop=True)
    duplicates_removed = before_dedup - len(df)

    if df["label"].nunique() != 2:
        raise ValueError("Both spam and ham emails are required.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nDataset preparation complete")
    print(f"Parsed emails:              {initial_count:,}")
    print(f"Conflicting rows removed:   {conflicting_rows:,}")
    print(f"Duplicate rows removed:     {duplicates_removed:,}")
    print(f"Skipped files:              {skipped:,}")
    print(f"Final emails:               {len(df):,}")
    print(f"Ham emails:                 {(df['label'] == 0).sum():,}")
    print(f"Spam emails:                {(df['label'] == 1).sum():,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()