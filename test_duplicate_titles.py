"""
Check data/<corpus>/merged.csv for duplicate titles.

Two records are flagged as duplicates if their titles match after light
normalization (lowercased, accents stripped, punctuation removed, whitespace
collapsed). For each duplicate group, the matching entries are printed with
year, sources, DOI, and ArXiv ID so you can decide whether they are truly
the same paper.

Exit code is non-zero if any duplicates are found, so this can be wired into
CI as a regression check after re-merging.

Run:
    python test_duplicate_titles.py                       # default: ai-sycophancy
    python test_duplicate_titles.py --corpus sycophancy
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from merge_data import CORPORA, norm_text

DATA_ROOT = Path(__file__).parent / "data"


def find_duplicate_titles(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = norm_text(row.get("title", ""))
        if key:
            groups[key].append(row)
    return {k: v for k, v in groups.items() if len(v) > 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--corpus",
        choices=CORPORA,
        default="ai-sycophancy",
        help="which corpus to check (default: ai-sycophancy)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged_csv = DATA_ROOT / args.corpus / "merged.csv"
    if not merged_csv.exists():
        print(f"missing {merged_csv} — run merge_data.py first", file=sys.stderr)
        return 2

    with merged_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dupes = find_duplicate_titles(rows)

    print(f"Loaded {len(rows)} records from {merged_csv}")
    if not dupes:
        print("No duplicate titles found.")
        return 0

    n_groups = len(dupes)
    n_extra = sum(len(v) - 1 for v in dupes.values())
    print(f"Found {n_groups} duplicate title group(s) covering {n_extra} extra record(s):\n")

    for key, group in sorted(dupes.items(), key=lambda kv: -len(kv[1])):
        print(f"  '{group[0]['title']}'  ({len(group)} entries)")
        for r in group:
            year = r.get("year") or "?"
            sources = r.get("sources") or "?"
            doi = r.get("doi") or "—"
            arxiv = r.get("arxiv_id") or "—"
            authors = (r.get("authors") or "")[:60]
            print(f"    - [{year}] sources={sources} doi={doi} arxiv={arxiv}")
            print(f"        authors: {authors}")
        print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
