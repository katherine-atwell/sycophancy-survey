"""
Analyze a merged paper dataset.

Reads data/<corpus>/merged.csv (produced by merge_data.py) and writes
summary + plots into results/<corpus>/. With --include-news, also reads
data/<corpus>/merged_with_news.csv and produces two additional result
sets:
  results/<corpus>/with-news/   — scholarly + news combined
  results/<corpus>/news-only/   — news rows only

Outputs in each result dir:
  - summary.txt: counts, top venues, top-cited papers
  - papers_per_year.png: bar chart of papers published per year
  - citations_per_year.png: total citations of papers published per year
  - cumulative_papers.png: cumulative paper count over time

Run:
    python analyze_results.py                          # default: ai-sycophancy
    python analyze_results.py --corpus sycophancy
    python analyze_results.py --include-news
"""

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import squarify
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).parent
CORPORA = ("ai-sycophancy", "sycophancy")


def load(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise SystemExit(f"missing {results_path} — run merge_data.py first")
    df = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    # Coerce types and drop rows with no year (can't plot them).
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def is_news(df: pd.DataFrame) -> pd.Series:
    return df["sources"].fillna("").str.contains("news", case=False)


def filter_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Keep papers where 'sycophan' literally appears in title or abstract.

    The S2 bulk-search matcher is fuzzy, so this trims false positives. News
    rows are spared — they came from a topic-targeted library search rather
    than a fuzzy keyword match, and their abstracts are usually empty.
    """
    text = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.lower()
    keep = text.str.contains("sycophan", regex=False) | is_news(df)
    dropped = (~keep).sum()
    if dropped:
        print(f"Dropping {dropped} records that don't literally mention 'sycophan*' in title/abstract")
    return df[keep].copy()


def write_summary(df: pd.DataFrame, out: Path, has_citations: bool = True) -> None:
    lines: list[str] = []
    lines.append(f"Total papers: {len(df)}")
    lines.append(f"Year range: {df['year'].min()}–{df['year'].max()}")
    if has_citations:
        lines.append(f"Total citations across all papers: {df['citation_count'].sum()}")
        lines.append(f"Median citations: {df['citation_count'].median():.1f}")
    lines.append("")

    lines.append("Papers per year:")
    for year, n in df.groupby("year").size().sort_index().items():
        lines.append(f"  {year}: {n}")
    lines.append("")

    lines.append("Top 15 venues:")
    for v, n in df["venue"].value_counts().head(15).items():
        if v:
            lines.append(f"  {n:>3}  {v}")
    lines.append("")

    if has_citations:
        lines.append("Top 20 most-cited papers:")
        top = df.sort_values("citation_count", ascending=False).head(20)
        for _, row in top.iterrows():
            lines.append(f"  [{row['citation_count']:>4} cites, {row['year']}] {row['title']}")
            if row.get("doi"):
                lines.append(f"        doi: {row['doi']}")

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def plot_papers_per_year(df: pd.DataFrame, out: Path, label: str) -> None:
    counts = df.groupby("year").size().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color="#3b6fb6")
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.set_title(f"Papers mentioning {label} per year")
    ax.set_xticks(counts.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_papers_per_year_line(df: pd.DataFrame, out: Path, label: str) -> None:
    counts = df.groupby("year").size().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(counts.index, counts.values, marker="o", color="#3b6fb6", linewidth=2)
    ax.fill_between(counts.index, counts.values, alpha=0.15, color="#3b6fb6")
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.set_title(f"Papers mentioning {label} per year")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_citations_per_year(df: pd.DataFrame, out: Path, label: str) -> None:
    cites = df.groupby("year")["citation_count"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cites.index, cites.values, color="#c2511f")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Total citations (as of scrape date)")
    ax.set_title(f"Citations to {label} papers, by publication year")
    ax.set_xticks(cites.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_combined(df: pd.DataFrame, out: Path, label: str) -> None:
    """Papers (bars) and citations (line) on the same time axis."""
    by_year = df.groupby("year").agg(papers=("title", "count"),
                                     citations=("citation_count", "sum")).sort_index()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(by_year.index, by_year["papers"], color="#3b6fb6", label="Papers")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Papers", color="#3b6fb6")
    ax1.tick_params(axis="y", labelcolor="#3b6fb6")
    ax1.set_xticks(by_year.index)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(by_year.index, by_year["citations"], color="#c2511f", marker="o", label="Citations")
    ax2.set_ylabel("Total citations", color="#c2511f")
    ax2.tick_params(axis="y", labelcolor="#c2511f")

    ax1.set_title(f"{label}: papers and citations over time")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# Venue-keyword rules for records whose `fields_of_study` is empty
# (typically WoS-only rows). First match wins, so ordering matters:
# unambiguous CS conferences first, then medicine/psych, broad CS last.
VENUE_RULES: list[tuple[str, list[str]]] = [
    ("Computer Science", [
        "arxiv", "zenodo", "aaai", "acl ", "naacl", "emnlp", "neurips", "icml",
        "iclr", "chi conference", "kdd", "ijcai", "natural language",
        "computational linguistic", "machine learning", "nlp",
    ]),
    ("Medicine", [
        "medic", "clinic", "hospit", "patient", "psychiatr", "surger", "diagnos",
        "gastroentero", "infection", "epidemiolog", "healthcare", "jmir",
        "annals of the academy of medicine", "open forum infectious",
        "research square", "medrxiv", "biorxiv",
    ]),
    ("Psychology", ["psycholog", "cognit", "mental", "neurosci", "bj psych"]),
    ("Education", [
        "educat", "pedagog", "librarian", "library",
        "trends in neuroscience and education",
    ]),
    ("Law", ["law", "legal", "jurispr"]),
    ("Economics", ["economic", "finance", "accounting", "business", "management"]),
    ("Sociology", [
        "social", "sociolog", "semiotic", "policy", "polit", "cultural",
        "anthropolog", "m/c journal", "humanit", "ssrn",
    ]),
    ("Philosophy", ["philosoph", "ethic", "foundations of science"]),
    ("Multidisciplinary", ["nature", "pnas", "royal society"]),
    ("Computer Science", [
        "computer", "software", "artificial intelligence", "ai &", "ai,",
        "linguistic", "electronic", "engineer", "workshop on",
        "information technology", "knowledge-based",
        "international journal of human-computer interaction",
        "international conference on artificial intelligence",
    ]),
]


def assign_domain(row: pd.Series) -> str:
    """Assign a single domain label per paper.

    S2's `fields_of_study` is authoritative when present (take the first
    listed field, which is S2's primary classification). Otherwise infer
    from venue with substring keyword rules; fall back to "Other".
    """
    fos = (row.get("fields_of_study") or "").strip()
    if fos:
        return fos.split(";")[0].strip()
    venue = (row.get("venue") or "").lower()
    if not venue:
        return "Other"
    for domain, keywords in VENUE_RULES:
        if any(kw in venue for kw in keywords):
            return domain
    return "Other"


def plot_domain_treemap(df: pd.DataFrame, out: Path, label: str) -> None:
    """Treemap of paper domains for the last 5 calendar years. News excluded."""
    current_year = date.today().year
    window = (current_year - 4, current_year)
    scholarly = df[~is_news(df)].copy()
    recent = scholarly[scholarly["year"].between(*window)].copy()
    if recent.empty:
        print(f"Skipping {out.name}: no scholarly records in {window[0]}–{window[1]}")
        return

    recent["domain"] = recent.apply(assign_domain, axis=1)
    counts = recent["domain"].value_counts()
    total = int(counts.sum())

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(counts))]
    labels = [
        f"{name}\n{n} ({n / total:.0%})" for name, n in counts.items()
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    squarify.plot(
        sizes=counts.values, label=labels, color=colors,
        alpha=0.85, ax=ax, pad=True,
        text_kwargs={"fontsize": 10, "color": "black"},
    )
    ax.set_title(
        f"Domains of {label} research papers, {window[0]}–{window[1]} "
        f"(n={total})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cumulative(df: pd.DataFrame, out: Path, label: str) -> None:
    counts = df.groupby("year").size().sort_index().cumsum()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(counts.index, counts.values, marker="o", color="#2a8c5b")
    ax.fill_between(counts.index, counts.values, alpha=0.2, color="#2a8c5b")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative papers")
    ax.set_title(f"Cumulative {label} papers over time")
    ax.set_xticks(counts.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


CORPUS_LABELS = {
    "ai-sycophancy": "AI sycophancy",
    "sycophancy": "sycophancy",
}


def write_results(df: pd.DataFrame, out_dir: Path, label: str, has_citations: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting outputs to {out_dir}\n")
    write_summary(df, out_dir / "summary.txt", has_citations)
    plot_papers_per_year(df, out_dir / "papers_per_year.png", label)
    plot_papers_per_year_line(df, out_dir / "papers_per_year_line.png", label)
    plot_cumulative(df, out_dir / "cumulative_papers.png", label)
    plot_domain_treemap(df, out_dir / "domain_treemap.png", label)
    if has_citations:
        plot_citations_per_year(df, out_dir / "citations_per_year.png", label)
        plot_combined(df, out_dir / "papers_and_citations.png", label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--corpus",
        choices=CORPORA,
        default="ai-sycophancy",
        help="which corpus to analyze (default: ai-sycophancy)",
    )
    parser.add_argument(
        "--include-news",
        action="store_true",
        help="also produce with-news/ and news-only/ result sets from "
             "data/<corpus>/merged_with_news.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    corpus_dir = HERE / "data" / args.corpus
    results_root = HERE / "results" / args.corpus
    label = CORPUS_LABELS[args.corpus]

    df = load(corpus_dir / "merged.csv")
    print(f"Loaded {len(df)} scholarly records with year info")
    df = filter_strict(df)
    print(f"{len(df)} after strict filter")
    write_results(df, results_root, label)

    if args.include_news:
        df_all = load(corpus_dir / "merged_with_news.csv")
        print(f"\nLoaded {len(df_all)} scholarly+news records with year info")
        df_all = filter_strict(df_all)
        print(f"{len(df_all)} after strict filter (news rows spared)")
        write_results(df_all, results_root / "with-news", f"{label} (incl. news)")

        df_news = df_all[is_news(df_all)].copy()
        print(f"\n{len(df_news)} news-only records")
        write_results(df_news, results_root / "news-only", f"{label} in news media",
                      has_citations=False)
