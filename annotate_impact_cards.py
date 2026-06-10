"""
Generate a Sycophancy Impact Card for each paper, using the framework from
"A Construct Validity Crisis in AI Sycophancy Evaluation" (§6.1, Figure 2).

For each paper (title + abstract + url) Claude returns a JSON object with
the impact-card fields:
  - behaviors_measured        (list[str])
  - referent                  (nested: position / person, each with present + subreferents + role)
  - norms_displaced           (list[str])
  - outcomes_measured         (list[str])
  - time_horizon              (one of: Immediate / Short-term / Medium-term / Long-term / None)
  - population                (str)
  - evidence_level            (one of the 5 ladder levels, or None)
  - proposed_mitigations      (list[str])
  - mitigation_outcomes       (str)
  - notes                     (str)

Output is JSONL — one impact card per line — written incrementally so a
crash or rate-limit abort doesn't lose progress. Re-running picks up where
it left off (keyed by paper_id, same scheme as annotate_papers.py).

Input file may be .json (a list of paper records) or .csv. Default is
data/ai-sycophancy/merged.json.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python annotate_impact_cards.py                                     # default: data/ai-sycophancy/merged.json
    python annotate_impact_cards.py --input data/ai-sycophancy/merged.csv
    python annotate_impact_cards.py --limit 5                           # smoke-test on 5 papers
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from tqdm import tqdm

import anthropic

HERE = Path(__file__).parent
MODEL = "claude-sonnet-4-6"

TIME_HORIZONS = ["Immediate", "Short-term", "Medium-term", "Long-term", "None"]
EVIDENCE_LEVELS = [
    "Behavioral detection",
    "Norm displacement",
    "Interaction impact",
    "Repeated interaction impact",
    "Institutional/societal impact",
    "None",
]
POSITION_SUBREFERENTS = ["Verifiable", "Subjective", "Both", "None"]
PERSON_SUBREFERENTS = ["Traits", "Emotions", "Both", "None"]
REFERENT_ROLES = ["primary", "secondary", "none"]

SCHEMA = {
    "type": "object",
    "properties": {
        "behaviors_measured": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Approval-seeking behaviors evaluated in the paper.",
        },
        "referent": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "subreferent": {"type": "string", "enum": POSITION_SUBREFERENTS},
                        "role": {"type": "string", "enum": REFERENT_ROLES},
                        "notes": {"type": "string"},
                    },
                    "required": ["present", "subreferent", "role", "notes"],
                    "additionalProperties": False,
                },
                "person": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "subreferent": {"type": "string", "enum": PERSON_SUBREFERENTS},
                        "role": {"type": "string", "enum": REFERENT_ROLES},
                        "notes": {"type": "string"},
                    },
                    "required": ["present", "subreferent", "role", "notes"],
                    "additionalProperties": False,
                },
            },
            "required": ["position", "person"],
            "additionalProperties": False,
        },
        "norms_displaced": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Context-relevant norms found to be displaced (accuracy, calibration, autonomy, etc.).",
        },
        "outcomes_measured": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific outcomes for which norm displacement was observed.",
        },
        "time_horizon": {"type": "string", "enum": TIME_HORIZONS},
        "population": {"type": "string"},
        "evidence_level": {"type": "string", "enum": EVIDENCE_LEVELS},
        "proposed_mitigations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "mitigation_outcomes": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": [
        "behaviors_measured",
        "referent",
        "norms_displaced",
        "outcomes_measured",
        "time_horizon",
        "population",
        "evidence_level",
        "proposed_mitigations",
        "mitigation_outcomes",
        "notes",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """\
You are an expert research assistant building **Sycophancy Impact Cards** for papers about sycophancy in AI/LLM systems, using the framework from "A Construct Validity Crisis in AI Sycophancy Evaluation" (Section 6.1).

The framework defines AI sycophancy as **user-approval-driven norm displacement**: a model behavior that accommodates a user's belief, affect, identity, goal, or framing in order to preserve approval or relational ease, while reducing fidelity to a context-relevant norm such as truth, calibration, autonomy, care, fairness, task quality, or institutional integrity.

For each paper (title + abstract + url) you receive, return a JSON object with the impact-card fields below.

## 1. behaviors_measured (array of strings)
The specific approval-seeking behaviors the paper observes or evaluates in LLMs. Use short, concrete labels — one behavior per array entry. Examples:
- "Opinion conformity"
- "Position switching"
- "Answer bias"
- "Error mimicry"
- "Excessive flattery"
- "Social validation"
- "Lack of pushback"
- "Uncritical emotional validation"
- "Hedging"
- "Emotional mimicry"

Return an empty array [] if the paper does not measure any specific behavior (e.g. a pure position paper).

## 2. referent (nested object)
From Ye et al. (2026): sycophantic behaviors are expressed toward either the user's **Position** (their expressed view/answer) or the **Person** themselves. Both can apply.

Sub-referents:
- Position: "Verifiable" (objective task with ground truth) vs. "Subjective" (opinions, advice, preferences) vs. "Both" vs. "None"
- Person:   "Traits" (praising character/abilities) vs. "Emotions" (validating feelings) vs. "Both" vs. "None"

Return:
{
  "position": {
    "present": true|false,
    "subreferent": "Verifiable" | "Subjective" | "Both" | "None",
    "role": "primary" | "secondary" | "none",
    "notes": "short phrase, e.g. 'position switching on MCQA'"
  },
  "person": {
    "present": true|false,
    "subreferent": "Traits" | "Emotions" | "Both" | "None",
    "role": "primary" | "secondary" | "none",
    "notes": "short phrase, e.g. 'flattery, affirming tone'"
  }
}

`role` indicates whether that referent type is the paper's main focus ("primary"), an additional/secondary concern ("secondary"), or absent ("none"). If `present` is false, set `subreferent` to "None" and `role` to "none".

## 3. norms_displaced (array of strings)
The specific context-relevant norms found (or claimed) to be displaced. Examples:
- "Accuracy"
- "Factuality"
- "Calibration"
- "Evidential independence"
- "Rationality" (Bayesian)
- "Epistemic accuracy"
- "Social judgment"
- "Moral autonomy"
- "Cognitive autonomy"
- "Fairness"
- "Institutional integrity"

Return [] if the paper does not assess any norm displacement (e.g. behavioral detection only, or pure position paper).

## 4. outcomes_measured (array of strings)
Specific outcomes the paper actually measures. Be concrete — name the metric, task, or behavioral signal. Examples:
- "Change in QA accuracy when users signal incorrect prior beliefs"
- "Rate of position switches when challenged"
- "Willingness to repair relationships"
- "User confidence in conversation outcome"
- "Attitude extremity after political discussion"
- "MMLU accuracy under user pressure"

Return [] if no quantitative outcome is measured.

## 5. time_horizon (string, exactly one of)
- "Immediate" — impact on a single output / single response quality
- "Short-term" — impact within one conversation, or multiple conversations over minutes to hours
- "Medium-term" — impact over days to weeks (multiple sessions)
- "Long-term" — impact over months or longer
- "None" — paper does not study an outcome with a clear time horizon (e.g. pure survey/position paper)

For most sycophancy benchmarks measuring single-response accuracy degradation, this is "Immediate". For studies running an experiment within one conversation (e.g. Cheng et al. 2026, Rathje et al. 2025), this is "Short-term".

## 6. population (string)
A concise description of the population studied. Include sample size if reported, the recruitment channel if notable, and whether participants are real users vs. simulated/prompt-manipulated users. Examples:
- "Simulated users via prompt manipulation; no human participants."
- "N=2,405 general adult users recruited via online panels."
- "Crowdworkers (Prolific) for preference judgments only."
- "Practicing clinicians (n=12) in a medical-QA setting."

Return "None" if the paper does not involve a study population (e.g. purely theoretical).

## 7. evidence_level (string, exactly one of)
The Evidence Maturity Ladder from §6.2:
- "Behavioral detection" — identifies an approval-seeking behavior but does NOT quantify norm displacement or downstream impact
- "Norm displacement" — measures the extent to which a context-relevant norm is displaced in the model's output (e.g. accuracy drop under user pressure)
- "Interaction impact" — measures impact on humans within a single interaction (e.g. Cheng et al. 2026, Rathje et al. 2025)
- "Repeated interaction impact" — measures impact across multiple conversations / sessions over time
- "Institutional/societal impact" — measures larger-scale impacts on institutions or society
- "None" — does not apply (e.g. pure position/survey paper)

Pick the **highest** level the paper actually reaches with evidence. A paper that both detects a behavior and measures accuracy drop is "Norm displacement", not "Behavioral detection".

## 8. proposed_mitigations (array of strings)
Mitigation techniques the paper proposes or tests. Be specific — name the method. Examples:
- "Best-of-N sampling with non-sycophancy-incentivizing preference model"
- "Supervised fine-tuning on synthetic non-sycophantic responses"
- "Activation steering along a sycophancy direction"
- "System-prompt edit instructing the model to disagree when warranted"

Return [] if no mitigation is proposed or tested.

## 9. mitigation_outcomes (string)
A concise summary (under 300 chars) of how the proposed mitigations performed. Return "N/A — no mitigations evaluated." if none were tested. Return "None" if mitigations were proposed but outcomes were not reported in the abstract.

## 10. notes (string)
Any other context that an impact-card reader should know — caveats, domain (medicine/law/education), notable scale, etc. Keep under 300 chars. Return "" (empty string) if there is nothing to add.

## Output rules
- Return ONLY the JSON object with exactly these ten top-level keys.
- For array fields, return [] (not null, not ["None"]) when nothing applies.
- For string-enum fields, use one of the allowed values verbatim (case- and spelling-exact).
- Be terse and grounded. Base annotations only on what the title, abstract, and (if accessible) full text actually say. If URLs exist, you may use them. Do not invent details — when the abstract is silent, prefer the more conservative classification (e.g. "Behavioral detection" over "Norm displacement", empty arrays over speculation).
"""


def paper_key(row: dict) -> str:
    """Stable identifier for dedup. Prefer real IDs over title hashes."""
    for k in ("doi", "arxiv_id", "s2_paper_id", "wos_id", "pubmed_id"):
        v = (row.get(k) or "").strip()
        if v:
            return f"{k}:{v}"
    title = (row.get("title") or "").strip().lower()
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


def load_rows(input_path: Path) -> list[dict]:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        with input_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            sys.exit(f"{input_path} must contain a JSON array of paper records")
        return data
    if suffix == ".csv":
        with input_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    sys.exit(f"unsupported input extension {suffix!r}; use .json or .csv")


def annotate_paper(
    client: anthropic.Anthropic, title: str, abstract: str, url: str
) -> tuple[dict, anthropic.types.Usage]:
    user_text = (
        f"Title: {title}\n\n"
        f"Abstract: {abstract or '(no abstract available)'}\n\n"
        f"URL: {url or '(no URL)'}\n\n"
        "Build the Sycophancy Impact Card for this paper according to the system prompt instructions."
    )
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
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "data" / "ai-sycophancy" / "merged.json",
        help="input file (.json list or .csv); default: data/ai-sycophancy/merged.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output jsonl (default: <input-dir>/impact_cards.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="annotate at most N papers (useful for smoke-testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or (input_path.parent / "impact_cards.jsonl")

    if not input_path.exists():
        sys.exit(f"missing {input_path}")

    rows = load_rows(input_path)
    existing = load_existing_keys(output_path)
    print(f"Input:  {input_path}  ({len(rows)} papers)")
    print(f"Output: {output_path}  ({len(existing)} already annotated)")

    todo = [r for r in rows if paper_key(r) not in existing]
    if args.limit:
        todo = todo[: args.limit]
    print(f"To do:  {len(todo)} paper(s)\n")
    if not todo:
        return

    client = anthropic.Anthropic()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = total_out = total_cache_read = total_cache_write = 0
    n_ok = n_err = 0

    with output_path.open("a", encoding="utf-8") as out_f:
        for row in tqdm(todo, total=len(todo)):
            key = paper_key(row)
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            url = (row.get("url") or "").strip()

            try:
                card, usage = annotate_paper(client, title, abstract, url)
            except anthropic.APIError as e:
                print(f"  error on {key}: {type(e).__name__}: {e}", file=sys.stderr)
                n_err += 1
                continue

            total_in += usage.input_tokens
            total_out += usage.output_tokens
            total_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
            total_cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0

            record = {
                "paper_id": key,
                "title": title,
                "authors": row.get("authors", ""),
                "year": row.get("year", ""),
                "venue": row.get("venue", ""),
                "doi": row.get("doi", ""),
                "arxiv_id": row.get("arxiv_id", ""),
                "url": url,
                "impact_card": card,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_ok += 1

    print(f"\nDone. {n_ok} annotated, {n_err} errored.")
    print(
        f"Tokens — input: {total_in}, output: {total_out}, "
        f"cache read: {total_cache_read}, cache write: {total_cache_write}"
    )


if __name__ == "__main__":
    main()
