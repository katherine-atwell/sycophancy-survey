"""
Scrape Semantic Scholar for articles about AI sycophancy.

Uses the Graph API bulk search endpoint, which supports boolean operators
and returns up to 1000 papers per page (with continuation tokens).

No API key required. If you have one, set it for higher rate limits:
    export S2_API_KEY="your-key-here"

Run:
    python scrape_semanticscholar.py

Output: s2_results.json and s2_results.csv
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

API_KEY = os.environ.get("S2_API_KEY")  # optional
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Bulk-search query syntax: + = AND, | = OR, parens for grouping, "" for phrases.
# Searches title + abstract.
AI_TERMS = [
    '"AI"',
    '"LLM"',
    '"LLMs"',
    '"large language models"',
    '"large language model"',
    '"generative AI"',
    '"artificial intelligence"',
    '"NLP"',
    '"natural language processing"',
    '"language models"',
    '"language model"',
]
ai_clause = " | ".join(AI_TERMS)
QUERY = (
    f'"AI sycophancy" | "LLM sycophancy" | '
    f"(({ai_clause}) + sycophancy)"
)

FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "authors",
    "year",
    "publicationDate",
    "venue",
    "publicationVenue",
    "externalIds",
    "citationCount",
    "referenceCount",
    "openAccessPdf",
    "url",
    "publicationTypes",
    "fieldsOfStudy",
])


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
def fetch_page(query: str, token: str | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    params = {"query": query, "fields": FIELDS}
    if token:
        params["token"] = token
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=60)
    if resp.status_code == 429:
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


def extract_record(p: dict) -> dict:
    authors = p.get("authors") or []
    author_names = "; ".join(a.get("name", "") for a in authors)
    ext = p.get("externalIds") or {}
    pub_venue = p.get("publicationVenue") or {}
    oa = p.get("openAccessPdf") or {}
    pub_types = p.get("publicationTypes") or []
    fos = p.get("fieldsOfStudy") or []

    return {
        "paperId": p.get("paperId", ""),
        "title": p.get("title", "") or "",
        "abstract": p.get("abstract", "") or "",
        "authors": author_names,
        "year": p.get("year", ""),
        "publication_date": p.get("publicationDate", "") or "",
        "venue": p.get("venue", "") or "",
        "publication_venue": pub_venue.get("name", "") if isinstance(pub_venue, dict) else "",
        "doi": ext.get("DOI", ""),
        "arxiv_id": ext.get("ArXiv", ""),
        "pubmed_id": ext.get("PubMed", ""),
        "corpus_id": ext.get("CorpusId", ""),
        "citation_count": p.get("citationCount", 0),
        "reference_count": p.get("referenceCount", 0),
        "open_access_pdf": oa.get("url", "") if isinstance(oa, dict) else "",
        "publication_types": "; ".join(pub_types),
        "fields_of_study": "; ".join(fos),
        "url": p.get("url", ""),
    }


def scrape_all() -> list[dict]:
    print(f"Query: {QUERY}\n")
    all_records: list[dict] = []
    token: str | None = None
    page = 0
    while True:
        page += 1
        print(f"Fetching page {page}" + (f" (token={token[:12]}...)" if token else "") + "...")
        data = fetch_page(QUERY, token)
        papers = data.get("data") or []
        total = data.get("total", 0)
        all_records.extend(extract_record(p) for p in papers)
        print(f"  got {len(papers)} papers (running total {len(all_records)} / {total})")

        token = data.get("token")
        if not token or not papers:
            break
        time.sleep(1)  # be polite even with API key

    return all_records


def save(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "s2_results.json"
    csv_path = out_dir / "s2_results.csv"

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")

    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    if not API_KEY:
        print("(no S2_API_KEY set — running unauthenticated, rate limit is lower)\n")
    records = scrape_all()
    print(f"\nTotal records: {len(records)}")
    save(records, Path(__file__).parent / "data")
