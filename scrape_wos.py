"""
Scrape Web of Science for articles about sycophancy.

Two corpora are supported:
  - ai-sycophancy: sycophancy co-occurring with AI/LLM/etc. (the original query)
  - sycophancy:    every paper mentioning sycophancy, no AI restriction

Uses the WoS Starter API. Set WOS_API_KEY in your environment before running:
    export WOS_API_KEY="your-key-here"
    python scrape_wos.py                       # default: ai-sycophancy
    python scrape_wos.py --corpus sycophancy

Output: data/<corpus>/wos_results.json and data/<corpus>/wos_results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

API_KEY = os.environ.get("WOS_API_KEY")
BASE_URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"
DB = "WOS"  # Web of Science Core Collection
PAGE_SIZE = 50  # Starter API max is 50

# AI-related terms; "sycophancy" must co-occur with at least one.
AI_TERMS = [
    "AI",
    "LLM",
    "LLMs",
    "large language models",
    "large language model",
    "generative AI",
    "artificial intelligence",
    "NLP",
    "natural language processing",
    "language models",
    "language model",
]

# TS= (Topic = title/abstract/keywords/keywords-plus) queries.
# The phrase clauses ("AI sycophancy", etc.) are technically subsumed by
# the (AI_TERMS) AND sycophancy clause, but kept explicit for clarity.
_ai_clause = " OR ".join(f'"{t}"' for t in AI_TERMS)

QUERIES = {
    "ai-sycophancy": (
        f'TS=("AI sycophancy" OR "LLM sycophancy" '
        f"OR (({_ai_clause}) AND sycophancy))"
    ),
    "sycophancy": 'TS=(sycophancy OR sycophantic OR sycophant)',
}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def fetch_page(query: str, page: int, limit: int = PAGE_SIZE) -> dict:
    """Fetch one page of results. Retries with backoff on transient failures."""
    headers = {"X-ApiKey": API_KEY, "Accept": "application/json"}
    params = {"db": DB, "q": query, "page": page, "limit": limit}
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    if resp.status_code == 429:
        # Rate limited — let tenacity retry after waiting.
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


def extract_record(hit: dict) -> dict:
    """Flatten a Starter API record into a row of interesting fields."""
    title = hit.get("title", "")
    uid = hit.get("uid", "")
    src = hit.get("source", {}) or {}
    names = hit.get("names", {}) or {}
    authors = names.get("authors", []) or []
    author_names = "; ".join(a.get("displayName", "") for a in authors)

    ids = hit.get("identifiers", {}) or {}
    links = hit.get("links", {}) or {}

    return {
        "uid": uid,
        "title": title,
        "authors": author_names,
        "source_title": src.get("sourceTitle", ""),
        "publish_year": src.get("publishYear", ""),
        "publish_month": src.get("publishMonth", ""),
        "volume": src.get("volume", ""),
        "issue": src.get("issue", ""),
        "pages": src.get("pages", {}).get("range", "") if isinstance(src.get("pages"), dict) else "",
        "doi": ids.get("doi", ""),
        "issn": ids.get("issn", ""),
        "eissn": ids.get("eissn", ""),
        "document_type": (hit.get("types") or [""])[0] if hit.get("types") else "",
        "wos_record_url": links.get("record", ""),
    }


def scrape_all(query: str) -> list[dict]:
    if not API_KEY:
        sys.exit("ERROR: set WOS_API_KEY in your environment before running.")

    print(f"Query: {query}\n")
    all_records: list[dict] = []
    page = 1
    while True:
        print(f"Fetching page {page}...")
        data = fetch_page(query, page)
        hits = data.get("hits", []) or []
        if not hits:
            break
        all_records.extend(extract_record(h) for h in hits)

        meta = data.get("metadata", {}) or {}
        total = meta.get("total", 0)
        print(f"  got {len(hits)} hits (running total {len(all_records)} / {total})")
        if len(all_records) >= total or len(hits) < PAGE_SIZE:
            break
        page += 1
        time.sleep(1)  # be polite

    return all_records


def save(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "wos_results.json"
    csv_path = out_dir / "wos_results.csv"

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")

    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--corpus",
        choices=sorted(QUERIES),
        default="ai-sycophancy",
        help="which query/corpus to scrape (default: ai-sycophancy)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records = scrape_all(QUERIES[args.corpus])
    print(f"\nTotal records: {len(records)}")
    save(records, Path(__file__).parent / "data" / args.corpus)
