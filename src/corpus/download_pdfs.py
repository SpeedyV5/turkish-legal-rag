from __future__ import annotations

from pathlib import Path
import time
import warnings

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SEED_PATH = Path("data/raw/mevzuat/seed_urls.csv")
PDF_DIR = Path("data/raw/mevzuat/pdfs")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_seed_csv(path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1254", "latin-1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] Seed CSV loaded with encoding: {enc}")
            return df
        except Exception as e:
            last_error = e

    raise ValueError(f"Could not read CSV with tried encodings. Last error: {last_error}")


def build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Connection": "keep-alive",
        }
    )
    return session


def is_probably_pdf(content: bytes, content_type: str | None) -> bool:
    if content.startswith(b"%PDF"):
        return True
    if content_type and "pdf" in content_type.lower():
        return True
    return False


def try_download(session: requests.Session, url: str, output_path: Path, verify: bool = True, timeout: int = 90) -> tuple[bool, str]:
    try:
        response = session.get(url, timeout=timeout, stream=True, verify=verify)
        response.raise_for_status()

        content = response.content
        content_type = response.headers.get("Content-Type", "")

        if not is_probably_pdf(content, content_type):
            return False, f"Response is not a PDF. Content-Type={content_type}"

        output_path.write_bytes(content)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def download_with_fallbacks(session: requests.Session, primary_url: str, fallback_url: str | None, output_path: Path) -> bool:
    attempts = [
        ("primary", primary_url, True),
        ("fallback", fallback_url, True) if fallback_url else None,
        ("primary_insecure", primary_url, False),
        ("fallback_insecure", fallback_url, False) if fallback_url else None,
    ]
    attempts = [a for a in attempts if a is not None and a[1]]

    for label, url, verify in attempts:
        print(f"[INFO] Trying {label}: {url}")
        if not verify:
            warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        ok, msg = try_download(session, url, output_path, verify=verify)
        if ok:
            print(f"[OK] Download succeeded via {label}")
            return True
        print(f"[WARN] {label} failed: {msg}")
        time.sleep(1.0)

    return False


def main() -> None:
    ensure_dir(PDF_DIR)

    if not SEED_PATH.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_PATH}")

    df = load_seed_csv(SEED_PATH)
    session = build_session()

    required_cols = {"doc_id", "title", "url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in seed file: {missing}")

    for _, row in df.iterrows():
        doc_id = str(row["doc_id"]).strip()
        primary_url = str(row["url"]).strip()
        fallback_url = str(row["fallback_url"]).strip() if "fallback_url" in row and pd.notna(row["fallback_url"]) else None

        output_path = PDF_DIR / f"{doc_id}.pdf"

        if output_path.exists():
            print(f"[SKIP] Already exists: {output_path}")
            continue

        print(f"[INFO] Downloading {doc_id}")
        success = download_with_fallbacks(session, primary_url, fallback_url, output_path)

        if not success:
            print(f"[ERROR] Failed for {doc_id} after all attempts")

    print("[INFO] Download phase completed.")


if __name__ == "__main__":
    main()