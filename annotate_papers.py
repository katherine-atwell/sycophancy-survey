"""
Pseudo-label each paper in a merged.csv with Claude Sonnet.

For each paper (title + abstract) the model returns a JSON object with six
fields:
  - "Application area of interest"
  - "Definition of sycophancy"
  - "How they are measuring sycophancy"
  - "Language models tested"
  - "Tested methods for mitigating sycophancy"
  - "novel evaluation metrics"

Empty fields are returned as the literal string "None". Output is JSONL —
one JSON object per line — written incrementally so a crash or rate-limit
abort does not lose progress. Re-running picks up where it left off.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python annotate_papers.py                                  # default: ai-sycophancy/merged.csv
    python annotate_papers.py --corpus sycophancy
    python annotate_papers.py --input data/ai-sycophancy/merged_with_news.csv
    python annotate_papers.py --limit 5                        # smoke-test on 5 papers
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import anthropic

HERE = Path(__file__).parent
CORPORA = ("ai-sycophancy")
MODEL = "claude-sonnet-4-6"

CATEGORIES = [
    "Application area of interest",
    "Definition of sycophancy",
    "How they are measuring sycophancy",
    "Language models tested",
    "Tested methods for mitigating sycophancy",
    "novel evaluation metrics",
]

SCHEMA = {
    "type": "object",
    "properties": {c: {"type": "string"} for c in CATEGORIES},
    "required": CATEGORIES,
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are an expert research assistant annotating papers about sycophancy in AI/LLM systems. Sycophancy is the tendency for an AI model to tell users what they want to hear — agreeing with their (often incorrect) claims, flipping opinions under social pressure, flattering users, or otherwise prioritizing apparent user approval over truthful or accurate responses.

For each paper you receive (title + abstract), output a JSON object with these six string fields:

## 1. "Application area of interest"
A specific application domain the paper focuses on. Examples: Medicine, Healthcare, Education, Tutoring, Law, Customer service, Mental health, Therapy, Theology, Religion, Scientific research, Politics, Finance, Software engineering.

Return the most specific domain that fits the paper's focus. If the paper studies sycophancy in general — not specific to any application area — or only evaluates on generic NLP benchmarks (e.g. TruthfulQA, MMLU, MT-Bench) without a domain focus, return exactly "None". Note that subject matter that appears in a benchmark (math, coding, trivia) is not by itself an "application area" — application areas are real-world domains where an AI is deployed to users.

Examples:
- "Sycophancy in clinical decision support tools" → "Medicine"
- "Investigating sycophantic behavior in LLMs" (no domain focus) → "None"
- "How GPT-4 flatters users on math word problems" → "None" (math is not an application domain)
- "LLM tutors and the tendency to agree with student mistakes" → "Education"

## 2. "Definition of sycophancy"
The paper's working definition of sycophancy, paraphrased in one concise sentence (under 200 characters). Return "None" if the paper does not define sycophancy.

Examples:
- "Adapting responses to match a user's perceived beliefs or preferences even at the cost of accuracy."
- "Agreement with the user's stated position regardless of its correctness."
- "Behavior in which a model abandons a correct answer when challenged by the user."

## 3. "How they are measuring sycophancy"
The concrete method, metric, or benchmark the paper uses to measure sycophancy. Be specific. Return "None" if the paper does not directly measure sycophancy (e.g. a pure position paper or survey).

Examples:
- "Rate of opinion flips when the user pushes back with disagreement."
- "Agreement rate with false premises embedded in user prompts."
- "Accuracy drop on math problems when the user states a wrong belief."
- "SycophancyEval benchmark from Sharma et al. (2023)."

## 4. "Language models tested"
A comma-separated list of every language model the paper evaluates for sycophancy. Use the model names as the paper refers to them, including size/version when given. Return "None" if the paper does not test specific models (e.g. a position paper, survey, or purely theoretical work).

Examples:
- "GPT-4, GPT-3.5-turbo, Claude 2, LLaMA-2-70B-Chat, PaLM 2"
- "Claude 3.5 Sonnet, Gemini 1.5 Pro"
- "LLaMA-2-7B-Chat, Mistral-7B-Instruct"
- "GPT-4o"

## 5. "Tested methods for mitigating sycophancy"
Any mitigation technique the paper actually tests: prompt engineering, fine-tuning, RLHF/DPO modifications, activation steering, decoding interventions, system-prompt edits, etc. Be specific. Return "None" if the paper only measures or analyzes sycophancy without testing any mitigation.

Examples:
- "Supervised fine-tuning on synthetic non-sycophantic responses."
- "Activation steering along a sycophancy direction identified via probing."
- "Adding 'be honest and disagree when warranted' to the system prompt."
- "DPO with preference pairs that reward disagreement when the user is wrong."

## 6. "novel evaluation metrics"
Any new evaluation metric the paper introduces (not just adopts from prior work). Return "None" if the paper only uses existing metrics.

Examples:
- "Sycophancy-Adjusted Accuracy (SAA)."
- "Belief Stability Score under adversarial user pushback."
- "Flip rate × confidence weighting."

## Output rules
- Return only the JSON object with exactly these six keys.
- Use the literal string "None" (not null, not an empty string) for categories that do not apply.
- Keep each value under 200 characters. Be terse and concrete.
- Base annotations only on what the title, abstract, and (if accessible) full text actually say. If URLs exist, try to access them and read the full article text if information is not in the abstract. Do not invent details.
"""


def paper_key(row: dict) -> str:
    """Stable identifier for dedup. Prefer real IDs over title hashes."""
    for k in ("doi", "arxiv_id", "s2_paper_id", "wos_id", "pubmed_id"):
        v = row.get(k, "").strip()
        if v:
            return f"{k}:{v}"
    title = row.get("title", "").strip().lower()
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
    return f"title:{h}"


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "paper_id" in rec:
                keys.add(rec["paper_id"])
    return keys


def annotate_paper(client: anthropic.Anthropic, title: str, abstract: str) -> tuple[dict, anthropic.types.Usage]:
    user_text = f"Title: {title}\n\nAbstract: {abstract or '(no abstract available — annotate from title alone)'}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_text}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corpus", choices=CORPORA, default="ai-sycophancy",
                        help="which corpus's merged.csv to annotate (default: ai-sycophancy)")
    parser.add_argument("--input", type=Path, default=None,
                        help="explicit input csv (default: data/<corpus>/merged.csv)")
    parser.add_argument("--output", type=Path, default=None,
                        help="explicit output jsonl (default: data/<corpus>/annotations.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="annotate at most N papers (useful for smoke-testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = HERE / "data" / args.corpus
    input_path = args.input or corpus_dir / "merged.csv"
    output_path = args.output or corpus_dir / "annotations.jsonl"

    if not input_path.exists():
        sys.exit(f"missing {input_path} — run merge_data.py first")

    with input_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    existing = load_existing_keys(output_path)
    print(f"Input:  {input_path}  ({len(rows)} papers)")
    print(f"Output: {output_path}  ({len(existing)} already annotated)")

    todo = [r for r in rows if paper_key(r) not in existing]
    if args.limit:
        todo = todo[:args.limit]
    print(f"To do:  {len(todo)} paper(s)\n")
    if not todo:
        return

    client = anthropic.Anthropic()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = total_out = total_cache_read = total_cache_write = 0
    n_ok = n_err = 0

    with output_path.open("a", encoding="utf-8") as out_f:
        for i, row in enumerate(todo, 1):
            key = paper_key(row)
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()
            print(f"[{i}/{len(todo)}] {title[:80]}")

            try:
                annotations, usage = annotate_paper(client, title, abstract)
            except anthropic.APIError as e:
                print(f"  error: {type(e).__name__}: {e}", file=sys.stderr)
                n_err += 1
                continue

            total_in += usage.input_tokens
            total_out += usage.output_tokens
            total_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
            total_cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0

            record = {
                "paper_id": key,
                "title": title,
                "doi": row.get("doi", ""),
                "arxiv_id": row.get("arxiv_id", ""),
                **annotations,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_ok += 1

    print(f"\nDone. {n_ok} annotated, {n_err} errored.")
    print(f"Tokens — input: {total_in}, output: {total_out}, "
          f"cache read: {total_cache_read}, cache write: {total_cache_write}")


if __name__ == "__main__":
    main()
