"""
Word cloud of paper authors' verbatim definitions of sycophancy.

Reads data/<corpus>/annotations.jsonl, extracts the "Verbatim definition of
sycophancy" field across all papers, removes generic English stopwords and
all morphological variants of "sycophancy", and renders a word cloud.

Output: results/<corpus>/definition_wordcloud.png

Run:
    python plot_definition_wordcloud.py                       # default: ai-sycophancy
    python plot_definition_wordcloud.py --corpus ai-sycophancy
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

HERE = Path(__file__).parent
CORPORA = ("ai-sycophancy",)
FIELD = "Verbatim definition of sycophancy"

# Variants of the focal term — excluded so the cloud shows the *concepts*
# authors use to define sycophancy, not the word itself.
SYCOPHANCY_TERMS = {
    "sycophancy", "sycophant", "sycophants",
    "sycophantic", "sycophantically", "sycophancies",
}

# Additional custom stopwords to remove from the word cloud
CUSTOM_STOPWORDS = {"model", "models", "ai", "a.i.", "artificial", "intelligence", "language", "large"}


def collect_definitions(path: Path) -> list[str]:
    out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            v = (rec.get(FIELD) or "").strip()
            if v and v.lower() != "none":
                out.append(v)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corpus", choices=CORPORA, default="ai-sycophancy",
                        help="which corpus to visualize (default: ai-sycophancy)")
    parser.add_argument("--input", type=Path, default=None,
                        help="explicit annotations jsonl (default: data/<corpus>/annotations.jsonl)")
    parser.add_argument("--output", type=Path, default=None,
                        help="explicit output path (default: results/<corpus>/definition_wordcloud.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = args.input or HERE / "data" / args.corpus / "annotations.jsonl"
    out_path = args.output or HERE / "results" / args.corpus / "definition_wordcloud.png"

    if not jsonl_path.exists():
        raise SystemExit(f"missing {jsonl_path} — run annotate_papers.py first")

    definitions = collect_definitions(jsonl_path)
    print(f"Loaded {len(definitions)} non-empty verbatim definitions from {jsonl_path.name}")
    if not definitions:
        raise SystemExit("no usable definitions found — was the verbatim backfill run?")

    text = " ".join(definitions)
    stopwords = set(STOPWORDS) | SYCOPHANCY_TERMS | CUSTOM_STOPWORDS

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
        colormap="viridis",
        max_words=200,
        random_state=0,
    ).generate(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Concepts in authors' verbatim definitions of sycophancy "
        f"(n={len(definitions)} papers)",
        fontsize=12,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
