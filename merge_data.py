"""
Merge all .csv and .tsv files in data/<corpus>/ into a single deduplicated dataset.

Matches papers across sources using (in order):
  1. DOI (lowercased)
  2. ArXiv ID — including ArXiv IDs encoded as fake DOIs
     (e.g. WoS's `arxiv:2411.10156` or S2's `10.48550/arxiv.2411.10156`)
  3. Normalized title (lowercased, accents stripped, punctuation removed)

The title fingerprint is intentionally title-only: a preprint and its
published version often differ by year and author transliteration but share
the title verbatim. Rare false positives (unrelated papers with identical
titles) are surfaced by test_duplicate_titles.py — but since exact matches
are merged here, that test only flags collisions this script *missed*.

When two records match, fields are merged: non-empty values win, longer
abstracts win, and IDs from each source are preserved.

Run:
    python merge_data.py                       # default: ai-sycophancy
    python merge_data.py --corpus sycophancy

Output: data/<corpus>/merged.csv and data/<corpus>/merged.json
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "data"
CORPORA = ("ai-sycophancy", "sycophancy")
OUTPUT_BASENAME = "merged"

# Unified output schema. Order is the CSV column order.
UNIFIED_FIELDS = [
    "title",
    "authors",
    "year",
    "publication_date",
    "venue",
    "doi",
    "arxiv_id",
    "pubmed_id",
    "wos_id",
    "s2_paper_id",
    "abstract",
    "citation_count",
    "publication_types",
    "fields_of_study",
    "keywords",
    "url",
    "open_access_pdf",
    "sources",
]


def norm_text(s: str) -> str:
    """Normalize for fuzzy matching: lowercase, strip punctuation, collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi


def norm_arxiv(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    arxiv_id = arxiv_id.strip().lower()
    arxiv_id = re.sub(r"^arxiv:", "", arxiv_id)
    return arxiv_id


# DOIs that are really ArXiv pointers in disguise. WoS exports preprints with
# `arxiv:NNNN.NNNNN` in the DI column; S2 sometimes uses DataCite-minted
# `10.48550/arXiv.NNNN.NNNNN`. Both should be promoted to arxiv_id so a
# preprint record matches a record that already has arxiv_id set.
ARXIV_DOI_RE = re.compile(r"^(?:arxiv:|10\.48550/arxiv\.)(.+)$", re.IGNORECASE)


def promote_arxiv_doi(rec: dict) -> dict:
    """If doi is an ArXiv pointer, move it into arxiv_id and clear doi."""
    doi = rec.get("doi") or ""
    m = ARXIV_DOI_RE.match(doi.strip())
    if not m:
        return rec
    extracted = norm_arxiv(m.group(1))
    if extracted and not rec.get("arxiv_id"):
        rec["arxiv_id"] = extracted
    rec["doi"] = ""
    return rec


def load_s2_csv(path: Path) -> list[dict]:
    """Load a Semantic Scholar CSV (output of scrape_semanticscholar.py)."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "title": r.get("title", "").strip(),
            "authors": r.get("authors", "").strip(),
            "year": str(r.get("year", "")).strip(),
            "publication_date": r.get("publication_date", "").strip(),
            "venue": r.get("publication_venue") or r.get("venue", "") or "",
            "doi": norm_doi(r.get("doi", "")),
            "arxiv_id": norm_arxiv(r.get("arxiv_id", "")),
            "pubmed_id": r.get("pubmed_id", "").strip(),
            "wos_id": "",
            "s2_paper_id": r.get("paperId", "").strip(),
            "abstract": r.get("abstract", "").strip(),
            "citation_count": r.get("citation_count", "").strip(),
            "publication_types": r.get("publication_types", "").strip(),
            "fields_of_study": r.get("fields_of_study", "").strip(),
            "keywords": "",
            "url": r.get("url", "").strip(),
            "open_access_pdf": r.get("open_access_pdf", "").strip(),
            "sources": "semanticscholar",
        })
    return out


def load_wos_tsv(path: Path) -> list[dict]:
    """Load a Web of Science tab-delimited export.

    WoS uses 2-letter field tags as headers. The export has duplicate AU
    columns, so we read by position and keep the first non-empty occurrence
    of each tag per row.
    """
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        rows = list(reader)

    def get(row: list[str], tag: str) -> str:
        for i, h in enumerate(header):
            if h == tag and i < len(row) and row[i].strip():
                return row[i].strip()
        return ""

    out = []
    for row in rows:
        if not any(row):
            continue
        authors = get(row, "AU")
        title = get(row, "TI")
        year = get(row, "PY")
        # WoS keywords: DE = author keywords, ID = keywords plus
        kw_de = get(row, "DE")
        kw_id = get(row, "ID")
        keywords = "; ".join(k for k in (kw_de, kw_id) if k)
        out.append({
            "title": title,
            "authors": authors,
            "year": year,
            "publication_date": get(row, "PD"),
            "venue": get(row, "SO"),
            "doi": norm_doi(get(row, "DI")),
            "arxiv_id": "",
            "pubmed_id": get(row, "PM"),
            "wos_id": get(row, "UT"),
            "s2_paper_id": "",
            "abstract": get(row, "AB"),
            "citation_count": get(row, "TC"),
            "publication_types": get(row, "DT"),
            "fields_of_study": "",
            "keywords": keywords,
            "url": "",
            "open_access_pdf": "",
            "sources": "wos",
        })
    return out


def load_generic_csv(path: Path, delimiter: str) -> list[dict]:
    """Best-effort loader for unknown csv/tsv files: map any matching columns."""
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
    out = []
    for r in rows:
        rec = {k: "" for k in UNIFIED_FIELDS}
        for k, v in r.items():
            if not k:
                continue
            key = k.strip().lower().replace(" ", "_")
            if key in rec and v:
                rec[key] = str(v).strip()
        if rec.get("doi"):
            rec["doi"] = norm_doi(rec["doi"])
        if rec.get("arxiv_id"):
            rec["arxiv_id"] = norm_arxiv(rec["arxiv_id"])
        rec["sources"] = rec.get("sources") or path.stem
        out.append(rec)
    return out


NEWS_FILENAME_RE = re.compile(r"news|magazine|newspaper|onesearch", re.IGNORECASE)


def is_news_file(path: Path) -> bool:
    return bool(NEWS_FILENAME_RE.search(path.stem))


# Patterns tried in order to pull a year (and date when available) out of
# the messy "Is Part Of" strings OneSearch produces, e.g.
#   "The Wall Street journal. Eastern edition, 2025-06-30"
#   "The Bombay Times and Journal of Commerce (1838-1859), 1856-10-11, p.653"
_DATE_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_DATE_YM = re.compile(r"(\d{4})-(\d{1,2})\b")
_DATE_COMMA_YEAR = re.compile(r",\s*(\d{4})\b")
_DATE_BARE_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")


def extract_date(text: str) -> tuple[str, str]:
    """Return (year, publication_date). publication_date is ISO when possible."""
    if not text:
        return "", ""
    m = _DATE_ISO.search(text)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        return y, f"{y}-{mo}-{d}"
    m = _DATE_YM.search(text)
    if m:
        y, mo = m.group(1), m.group(2).zfill(2)
        return y, f"{y}-{mo}"
    m = _DATE_COMMA_YEAR.search(text) or _DATE_BARE_YEAR.search(text)
    return (m.group(1), "") if m else ("", "")


def _strip_trailing_date(text: str) -> str:
    return re.sub(r",\s*\d{4}(-\d{1,2})?(-\d{1,2})?.*$", "", text).strip()


def load_news_csv(path: Path) -> list[dict]:
    """Load a OneSearch-style newspaper/magazine export (ProQuest/Primo)."""
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        ipo = (r.get("Is Part Of") or "").strip()
        cd = (r.get("Creation Date") or "").strip()
        year, pubdate = extract_date(cd or ipo)
        venue = _strip_trailing_date(ipo) or (r.get("Publisher") or "").strip()
        authors = "; ".join(
            a.strip() for a in (r.get("Author", ""), r.get("Contributor", ""))
            if a and a.strip()
        )
        out.append({
            "title": (r.get("Title") or "").strip(),
            "authors": authors,
            "year": year,
            "publication_date": pubdate,
            "venue": venue,
            "doi": "",
            "arxiv_id": "",
            "pubmed_id": "",
            "wos_id": "",
            "s2_paper_id": "",
            "abstract": (r.get("Description") or "").strip(),
            "citation_count": "",
            "publication_types": "news",
            "fields_of_study": "",
            "keywords": (r.get("Subjects") or "").strip(),
            "url": (r.get("permalink") or "").strip(),
            "open_access_pdf": "",
            "sources": "news",
        })
    return out


def detect_and_load(path: Path) -> list[dict]:
    name = path.name.lower()
    if name.startswith("s2_results") and path.suffix == ".csv":
        return load_s2_csv(path)
    if name == "wos.tsv":
        return load_wos_tsv(path)
    if is_news_file(path):
        return load_news_csv(path)
    delim = "\t" if path.suffix == ".tsv" else ","
    return load_generic_csv(path, delim)


def merge_records(a: dict, b: dict) -> dict:
    """Merge b into a: non-empty fields from b fill blanks in a."""
    merged = dict(a)
    for k, v in b.items():
        if not v:
            continue
        if k == "sources":
            existing = set(s.strip() for s in (merged.get("sources") or "").split(";") if s.strip())
            for s in str(v).split(";"):
                if s.strip():
                    existing.add(s.strip())
            merged["sources"] = "; ".join(sorted(existing))
        elif not merged.get(k):
            merged[k] = v
        elif k == "abstract" and len(str(v)) > len(str(merged[k])):
            # Prefer the longer abstract
            merged[k] = v
    return merged


def dedupe(records: list[dict]) -> list[dict]:
    """Index by DOI, ArXiv, and normalized title; merge when any matches."""
    by_doi: dict[str, int] = {}
    by_arxiv: dict[str, int] = {}
    by_title: dict[str, int] = {}
    out: list[dict] = []

    for rec in records:
        rec = promote_arxiv_doi(rec)
        idx = None
        if rec.get("doi"):
            idx = by_doi.get(rec["doi"])
        if idx is None and rec.get("arxiv_id"):
            idx = by_arxiv.get(rec["arxiv_id"])
        if idx is None:
            title_key = norm_text(rec.get("title", ""))
            if title_key:
                idx = by_title.get(title_key)

        if idx is None:
            out.append(rec)
            idx = len(out) - 1
        else:
            out[idx] = merge_records(out[idx], rec)

        merged = out[idx]
        if merged.get("doi"):
            by_doi[merged["doi"]] = idx
        if merged.get("arxiv_id"):
            by_arxiv[merged["arxiv_id"]] = idx
        title_key = norm_text(merged.get("title", ""))
        if title_key:
            by_title[title_key] = idx

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--corpus",
        choices=CORPORA,
        default="ai-sycophancy",
        help="which corpus subfolder under data/ to merge (default: ai-sycophancy)",
    )
    return parser.parse_args()


def write_merged(records: list[dict], data_dir: Path, basename: str) -> None:
    print(f"\n{basename}: {len(records)} records before dedup")
    deduped = dedupe(records)
    print(f"{basename}: {len(deduped)} records after dedup")

    out_csv = data_dir / f"{basename}.csv"
    out_json = data_dir / f"{basename}.json"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        writer.writeheader()
        for rec in deduped:
            writer.writerow({k: rec.get(k, "") for k in UNIFIED_FIELDS})
    print(f"Wrote {out_csv}")
    out_json.write_text(json.dumps(deduped, indent=2, ensure_ascii=False))
    print(f"Wrote {out_json}")


def main() -> None:
    args = parse_args()
    data_dir = DATA_ROOT / args.corpus
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    files = sorted(
        p for p in data_dir.iterdir()
        if p.suffix.lower() in (".csv", ".tsv") and not p.stem.startswith(OUTPUT_BASENAME)
    )
    if not files:
        raise SystemExit(f"no .csv or .tsv files in {data_dir}")

    scholarly: list[dict] = []
    news: list[dict] = []
    for path in files:
        records = detect_and_load(path)
        bucket = "news" if is_news_file(path) else "scholarly"
        print(f"  [{bucket}] {path.name}: {len(records)} records")
        (news if bucket == "news" else scholarly).extend(records)

    write_merged(scholarly, data_dir, OUTPUT_BASENAME)
    if news:
        write_merged(scholarly + news, data_dir, f"{OUTPUT_BASENAME}_with_news")


if __name__ == "__main__":
    main()
