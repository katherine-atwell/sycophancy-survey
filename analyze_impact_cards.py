"""
Analyze the set of Sycophancy Impact Cards produced by annotate_impact_cards.py.

Reads data/<corpus>/impact_cards.jsonl and writes summary + plots into
results/<corpus>/impact_cards/.

Outputs:
  - summary.txt:                  counts and per-rung paper lists
  - evidence_level.png:           papers per rung of the Evidence Maturity Ladder
  - time_horizon.png:             papers per time-horizon bucket
  - referent_combo.png:           Position-only / Person-only / Both / Neither
  - referent_subreferents.png:    grouped bar — Position (Verifiable/Subjective)
                                  vs Person (Traits/Emotions)
  - top_behaviors.png:            top-15 behaviors_measured
  - top_norms.png:                top-15 norms_displaced
  - top_mitigations.png:          top-15 proposed_mitigations
  - evidence_x_time.png:          heatmap of evidence level × time horizon

Run:
    python analyze_impact_cards.py                         # default: ai-sycophancy
    python analyze_impact_cards.py --corpus sycophancy
    python analyze_impact_cards.py --top 20
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).parent
CORPORA = ("ai-sycophancy", "sycophancy")

EVIDENCE_ORDER = [
    "Behavioral detection",
    "Norm displacement",
    "Interaction impact",
    "Repeated interaction impact",
    "Institutional/societal impact",
    "None",
]
TIME_ORDER = ["Immediate", "Short-term", "Medium-term", "Long-term", "None"]
REFERENT_COMBOS = ["Position only", "Person only", "Both", "Neither"]

# Shared palette (coolors.co/243e36-ffaf87-74a57f-ff8e72-ed6a5e).
PALETTE = {
    "primary":   "#243E36",  # dark green — main bars, Position
    "secondary": "#74A57F",  # sage       — referent_combo, mitigations
    "accent":    "#ED6A5E",  # red-coral  — Person, time_horizon
    "warm":      "#FF8E72",  # coral      — norms
    "light":     "#FFAF87",  # peach      — highlights, treemap blends
}
PALETTE_ORDER = [
    PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
    PALETTE["warm"], PALETTE["light"],
]
HEATMAP_CMAP = sns.light_palette(PALETTE["primary"], as_cmap=True)

# Fonts are scaled for two-column paper rendering, so figures get shrunk
# downstream — bump font_scale and figsize so labels stay legible.
sns.set_theme(
    style="ticks",
    context="notebook",
    font_scale=2.0,
    palette=PALETTE_ORDER,
    rc={
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelweight": "medium",
        "savefig.bbox": "tight",
    },
)


def load_cards(path: Path) -> pd.DataFrame:
    """Flatten the JSONL into a one-row-per-paper DataFrame."""
    if not path.exists():
        raise SystemExit(f"missing {path} — run annotate_impact_cards.py first")

    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            card = rec.get("impact_card", {}) or {}
            ref = card.get("referent", {}) or {}
            pos = ref.get("position", {}) or {}
            per = ref.get("person", {}) or {}
            rows.append({
                "paper_id": rec.get("paper_id", ""),
                "title": rec.get("title", ""),
                "year": rec.get("year", ""),
                "venue": rec.get("venue", ""),
                "behaviors_measured": card.get("behaviors_measured") or [],
                "norms_displaced": card.get("norms_displaced") or [],
                "outcomes_measured": card.get("outcomes_measured") or [],
                "time_horizon": card.get("time_horizon") or "None",
                "population": card.get("population") or "",
                "evidence_level": card.get("evidence_level") or "None",
                "proposed_mitigations": card.get("proposed_mitigations") or [],
                "mitigation_outcomes": card.get("mitigation_outcomes") or "",
                "notes": card.get("notes") or "",
                "position_present": bool(pos.get("present")),
                "person_present": bool(per.get("present")),
                "position_subref": pos.get("subreferent") or "None",
                "person_subref": per.get("subreferent") or "None",
                "position_role": pos.get("role") or "none",
                "person_role": per.get("role") or "none",
            })
    df = pd.DataFrame(rows)

    def combo(r):
        if r["position_present"] and r["person_present"]:
            return "Both"
        if r["position_present"]:
            return "Position only"
        if r["person_present"]:
            return "Person only"
        return "Neither"

    df["referent_combo"] = df.apply(combo, axis=1)
    df["has_mitigations"] = df["proposed_mitigations"].map(lambda xs: len(xs) > 0)
    return df


def normalize_label(s: str) -> str:
    """Light normalization for free-text labels so 'Accuracy' == 'accuracy'."""
    return " ".join(s.strip().lower().split())


def count_listcol(df: pd.DataFrame, col: str) -> Counter:
    """Counter of normalized labels across a list-of-strings column."""
    c: Counter = Counter()
    for items in df[col]:
        for v in items:
            label = normalize_label(v)
            if label:
                c[label] += 1
    return c


def write_summary(df: pd.DataFrame, out: Path, top_n: int) -> None:
    lines: list[str] = []
    lines.append(f"Total impact cards: {len(df)}")
    lines.append("")

    lines.append("Evidence Level:")
    counts = df["evidence_level"].value_counts()
    for level in EVIDENCE_ORDER:
        n = int(counts.get(level, 0))
        pct = n / len(df) * 100 if len(df) else 0
        lines.append(f"  {n:>3}  ({pct:4.1f}%)  {level}")
    lines.append("")

    lines.append("Time horizon:")
    counts = df["time_horizon"].value_counts()
    for h in TIME_ORDER:
        n = int(counts.get(h, 0))
        lines.append(f"  {n:>3}  {h}")
    lines.append("")

    lines.append("Referent combo:")
    counts = df["referent_combo"].value_counts()
    for c in REFERENT_COMBOS:
        n = int(counts.get(c, 0))
        lines.append(f"  {n:>3}  {c}")
    lines.append("")

    lines.append("Position sub-referent (among Position-present papers):")
    for k, n in df[df["position_present"]]["position_subref"].value_counts().items():
        lines.append(f"  {n:>3}  {k}")
    lines.append("")

    lines.append("Person sub-referent (among Person-present papers):")
    for k, n in df[df["person_present"]]["person_subref"].value_counts().items():
        lines.append(f"  {n:>3}  {k}")
    lines.append("")

    n_mit = int(df["has_mitigations"].sum())
    lines.append(f"Papers proposing/testing mitigations: {n_mit} / {len(df)} "
                 f"({n_mit / len(df) * 100:.1f}%)")
    lines.append("")

    for col, header in [
        ("behaviors_measured", f"Top {top_n} behaviors_measured"),
        ("norms_displaced", f"Top {top_n} norms_displaced"),
        ("proposed_mitigations", f"Top {top_n} proposed_mitigations"),
    ]:
        lines.append(f"{header}:")
        for label, n in count_listcol(df, col).most_common(top_n):
            lines.append(f"  {n:>3}  {label}")
        lines.append("")

    lines.append("Sample papers at each evidence rung (up to 5 each):")
    for level in EVIDENCE_ORDER:
        sub = df[df["evidence_level"] == level]
        if sub.empty:
            continue
        lines.append(f"  -- {level} ({len(sub)}) --")
        for _, r in sub.head(5).iterrows():
            lines.append(f"    [{r['year']}] {r['title']}")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def plot_categorical(counts: pd.Series, order: list[str], title: str, out: Path,
                     xlabel: str, color: str = PALETTE["primary"]) -> None:
    values = [int(counts.get(k, 0)) for k in order]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(order, values, color=color)
    ax.set_ylabel("Papers")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i, v in enumerate(values):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=16)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_subreferents(df: pd.DataFrame, out: Path) -> None:
    pos = df[df["position_present"]]["position_subref"].value_counts()
    per = df[df["person_present"]]["person_subref"].value_counts()
    cats = ["Verifiable", "Subjective", "Both", "Traits", "Emotions"]
    vals = [
        int(pos.get("Verifiable", 0)), int(pos.get("Subjective", 0)), int(pos.get("Both", 0)),
        int(per.get("Traits", 0)), int(per.get("Emotions", 0)),
    ]
    colors = [PALETTE["primary"]] * 3 + [PALETTE["accent"]] * 2
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(cats, vals, color=colors)
    ax.set_ylabel("Papers")
    ax.set_title("Sub-referents: Position vs Person")
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=16)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["primary"], label="Position"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["accent"], label="Person"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_top(counter: Counter, title: str, out: Path, top_n: int,
             color: str = PALETTE["primary"]) -> None:
    items = counter.most_common(top_n)
    if not items:
        print(f"  (no data for {out.name}, skipping)")
        return
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.55 * len(labels) + 2)))
    ax.barh(labels, values, color=color)
    ax.set_xlabel("Papers")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=16)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_evidence_x_time(df: pd.DataFrame, out: Path) -> None:
    ct = pd.crosstab(df["evidence_level"], df["time_horizon"])
    ct = ct.reindex(index=EVIDENCE_ORDER, columns=TIME_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(ct, annot=True, fmt="d", cmap=HEATMAP_CMAP, cbar=False, ax=ax,
                linewidths=0.5, linecolor="white", annot_kws={"size": 16})
    ax.set_title("Evidence level × time horizon")
    ax.set_xlabel("Time horizon")
    ax.set_ylabel("Evidence level")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--corpus", choices=CORPORA, default="ai-sycophancy")
    p.add_argument("--input", type=Path, default=None,
                   help="explicit input jsonl (default: data/<corpus>/impact_cards.jsonl)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="explicit output dir (default: results/<corpus>/impact_cards/)")
    p.add_argument("--top", type=int, default=15,
                   help="top-N for list-field plots and summary tables (default: 15)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or HERE / "data" / args.corpus / "impact_cards.jsonl"
    outdir = args.outdir or HERE / "results" / args.corpus / "impact_cards"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_cards(input_path)
    print(f"Loaded {len(df)} impact cards from {input_path}\n")

    write_summary(df, outdir / "summary.txt", args.top)

    plot_categorical(
        df["evidence_level"].value_counts(), EVIDENCE_ORDER,
        "Evidence Level", outdir / "evidence_level.png", "Evidence level",
    )
    plot_categorical(
        df["time_horizon"].value_counts(), TIME_ORDER,
        "Time horizon of measured outcomes", outdir / "time_horizon.png", "Time horizon",
        color=PALETTE["accent"],
    )
    plot_categorical(
        df["referent_combo"].value_counts(), REFERENT_COMBOS,
        "Referent: Position vs Person", outdir / "referent_combo.png", "Referent",
        color=PALETTE["secondary"],
    )
    plot_subreferents(df, outdir / "referent_subreferents.png")

    plot_top(count_listcol(df, "behaviors_measured"),
             f"Top {args.top} behaviors measured", outdir / "top_behaviors.png", args.top)
    plot_top(count_listcol(df, "norms_displaced"),
             f"Top {args.top} norms displaced", outdir / "top_norms.png", args.top,
             color=PALETTE["warm"])
    plot_top(count_listcol(df, "proposed_mitigations"),
             f"Top {args.top} proposed mitigations", outdir / "top_mitigations.png", args.top,
             color=PALETTE["secondary"])

    plot_evidence_x_time(df, outdir / "evidence_x_time.png")


if __name__ == "__main__":
    main()
