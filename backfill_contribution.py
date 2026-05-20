"""
Backfill the "Contribution type" field on existing annotations.

Loads data/<corpus>/annotations.jsonl, then for each record that lacks the
field (or whose value is empty), looks up the original paper's title/abstract
from data/<corpus>/merged.csv and asks Claude Sonnet for *just* that field.
Records that already have a non-empty value are skipped — re-running is safe
and resumable.

The whole JSONL is rewritten atomically after every successful update, so a
crash or rate-limit interrupt won't corrupt the file or lose finished work.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python backfill_contribution.py                                   # default: ai-sycophancy
    python backfill_contribution.py --corpus ai-sycophancy --limit 5  # smoke-test
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import anthropic
from tqdm import tqdm

from annotate_papers import paper_key

HERE = Path(__file__).parent
CORPORA = ("ai-sycophancy",)
MODEL = "claude-sonnet-4-6"
FIELD = "Contribution type"

SCHEMA = {
    "type": "object",
    "properties": {FIELD: {"type": "string"}},
    "required": [FIELD],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""\
You are an expert research assistant. For each paper you receive (title + abstract) classify the primary type(s) of contribution it makes.

Choose **one or more** labels from this fixed list, copied exactly as written:

- "Survey / Position" — literature review, opinion piece, perspective, position paper, or purely conceptual/theoretical argument
- "Empirical findings" — new experimental results that characterize, measure, or analyze sycophancy
- "Novel benchmark" — introduces a new dataset, benchmark, or test suite for evaluating sycophancy
- "Novel metric" — introduces a new evaluation metric or scoring method for sycophancy
- "Novel mitigation method" — proposes a new method, intervention, training scheme, or steering technique to reduce sycophancy
- "Tool / software" — releases a tool, library, framework, or system (independent of empirical work)
- "Other" — none of the above clearly applies, or the contribution can't be determined from the available text

Pick every label that genuinely applies. Many papers will warrant two labels (e.g. an empirical paper that also introduces a benchmark). Join multiple labels with "; " (semicolon-space). Use the labels exactly as written above — do not invent new labels.

Examples:
- "Survey / Position"
- "Empirical findings"
- "Empirical findings; Novel benchmark"
- "Empirical findings; Novel mitigation method"
- "Novel benchmark; Novel metric"
- "Novel mitigation method; Tool / software"

Return only a JSON object with exactly one key: "{FIELD}".
"""


def needs_backfill(rec: dict) -> bool:
    v = rec.get(FIELD)
    return not v or not str(v).strip() or str(v).strip() == "None"


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl_atomic(records: list[dict], path: Path) -> None:
    """Write all records to a temp file in the same dir, then atomic-rename."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def build_merged_lookup(merged_csv: Path) -> dict[str, dict]:
    with merged_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {paper_key(r): r for r in rows}


def call_llm(client: anthropic.Anthropic, title: str, abstract: str, url: str) -> tuple[str, anthropic.types.Usage]:
    user_text = (
        f"Title: {title}\n\n"
        f"Abstract: {abstract or '(no abstract available)'}\n\n"
        f"URL: {url or '(no URL)'}\n\n"
        "Classify the contribution type(s) per the system prompt."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_text}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)[FIELD], response.usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corpus", choices=CORPORA, default="ai-sycophancy",
                        help="which corpus to backfill (default: ai-sycophancy)")
    parser.add_argument("--input", type=Path, default=None,
                        help="explicit annotations jsonl (default: data/<corpus>/annotations.jsonl)")
    parser.add_argument("--merged", type=Path, default=None,
                        help="explicit merged csv (default: data/<corpus>/merged.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="backfill at most N papers (useful for smoke-testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = HERE / "data" / args.corpus
    jsonl_path = args.input or corpus_dir / "annotations.jsonl"
    merged_csv = args.merged or corpus_dir / "merged.csv"

    if not jsonl_path.exists():
        sys.exit(f"missing {jsonl_path} — run annotate_papers.py first")
    if not merged_csv.exists():
        sys.exit(f"missing {merged_csv} — run merge_data.py first")

    records = load_jsonl(jsonl_path)
    merged_lookup = build_merged_lookup(merged_csv)
    todo_idx = [i for i, r in enumerate(records) if needs_backfill(r)]
    if args.limit:
        todo_idx = todo_idx[:args.limit]

    print(f"Input:    {jsonl_path}  ({len(records)} records)")
    print(f"Merged:   {merged_csv}  ({len(merged_lookup)} papers)")
    print(f"To fill:  {len(todo_idx)} record(s) missing '{FIELD}'\n")
    if not todo_idx:
        return

    client = anthropic.Anthropic()
    total_in = total_out = total_cache_read = total_cache_write = 0
    n_ok = n_err = n_missing_source = 0

    for i in tqdm(todo_idx):
        rec = records[i]
        key = rec.get("paper_id", "")
        source = merged_lookup.get(key)
        if source is None:
            print(f"  no merged.csv match for {key}, skipping", file=sys.stderr)
            n_missing_source += 1
            continue

        title = source.get("title", "").strip()
        abstract = source.get("abstract", "").strip()
        url = source.get("url", "").strip()

        try:
            value, usage = call_llm(client, title, abstract, url)
        except anthropic.APIError as e:
            print(f"  error on {key}: {type(e).__name__}: {e}", file=sys.stderr)
            n_err += 1
            continue

        total_in += usage.input_tokens
        total_out += usage.output_tokens
        total_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
        total_cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0

        rec[FIELD] = value
        write_jsonl_atomic(records, jsonl_path)
        n_ok += 1

    print(f"\nDone. {n_ok} filled, {n_err} errored, {n_missing_source} unmatched.")
    print(f"Tokens — input: {total_in}, output: {total_out}, "
          f"cache read: {total_cache_read}, cache write: {total_cache_write}")


if __name__ == "__main__":
    main()
