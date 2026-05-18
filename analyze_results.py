"""
Analyze the merged paper dataset.

Reads data/merged.csv (produced by merge_data.py) and writes:
  - summary.txt: counts, top venues, top-cited papers
  - papers_per_year.png: bar chart of papers published per year
  - citations_per_year.png: total citations of papers published per year
  - cumulative_papers.png: cumulative paper count over time

Run:
    python analyze_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
DATABASE = "merged"
RESULTS_PATH = HERE / "data" / "merged.csv"
OUT_DIR = HERE / "results" / DATABASE


def load() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"missing {RESULTS_PATH} — run merge_data.py first")
    df = pd.read_csv(RESULTS_PATH, dtype=str, keep_default_na=False)
    # Coerce types and drop rows with no year (can't plot them).
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def filter_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only papers where 'sycophan' literally appears in title or abstract.

    The S2 bulk-search matcher is fuzzy, so this trims false positives.
    """
    text = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.lower()
    keep = text.str.contains("sycophan", regex=False)
    dropped = (~keep).sum()
    if dropped:
        print(f"Dropping {dropped} records that don't literally mention 'sycophan*' in title/abstract")
    return df[keep].copy()


def write_summary(df: pd.DataFrame, out: Path) -> None:
    lines: list[str] = []
    lines.append(f"Total papers: {len(df)}")
    lines.append(f"Year range: {df['year'].min()}–{df['year'].max()}")
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

    lines.append("Top 20 most-cited papers:")
    top = df.sort_values("citation_count", ascending=False).head(20)
    for _, row in top.iterrows():
        lines.append(f"  [{row['citation_count']:>4} cites, {row['year']}] {row['title']}")
        if row.get("doi"):
            lines.append(f"        doi: {row['doi']}")

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def plot_papers_per_year(df: pd.DataFrame, out: Path) -> None:
    counts = df.groupby("year").size().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color="#3b6fb6")
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.set_title("Papers mentioning AI sycophancy per year")
    ax.set_xticks(counts.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_papers_per_year_line(df: pd.DataFrame, out: Path) -> None:
    counts = df.groupby("year").size().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(counts.index, counts.values, marker="o", color="#3b6fb6", linewidth=2)
    ax.fill_between(counts.index, counts.values, alpha=0.15, color="#3b6fb6")
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.set_title("Papers mentioning AI sycophancy per year")
    ax.set_xticks(counts.index)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_citations_per_year(df: pd.DataFrame, out: Path) -> None:
    cites = df.groupby("year")["citation_count"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cites.index, cites.values, color="#c2511f")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Total citations (as of scrape date)")
    ax.set_title("Citations to AI-sycophancy papers, by publication year")
    ax.set_xticks(cites.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_combined(df: pd.DataFrame, out: Path) -> None:
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

    ax1.set_title("AI sycophancy: papers and citations over time")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_cumulative(df: pd.DataFrame, out: Path) -> None:
    counts = df.groupby("year").size().sort_index().cumsum()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(counts.index, counts.values, marker="o", color="#2a8c5b")
    ax.fill_between(counts.index, counts.values, alpha=0.2, color="#2a8c5b")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative papers")
    ax.set_title("Cumulative AI sycophancy papers over time")
    ax.set_xticks(counts.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} records with year info")
    df = filter_strict(df)
    print(f"{len(df)} after strict filter\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {OUT_DIR}\n")

    write_summary(df, OUT_DIR / "summary.txt")
    plot_papers_per_year(df, OUT_DIR / "papers_per_year.png")
    plot_papers_per_year_line(df, OUT_DIR / "papers_per_year_line.png")
    plot_citations_per_year(df, OUT_DIR / "citations_per_year.png")
    plot_combined(df, OUT_DIR / "papers_and_citations.png")
    plot_cumulative(df, OUT_DIR / "cumulative_papers.png")
