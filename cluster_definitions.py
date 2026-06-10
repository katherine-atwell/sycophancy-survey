"""
Cluster authors' verbatim definitions of AI sycophancy to surface divergence
and any discrete clusters.

Reads data/<corpus>/annotations.jsonl, extracts the "Verbatim definition of
sycophancy" field, embeds each definition with one of three methods, then:

  - measures pairwise cosine similarity (the "how much do they diverge" picture)
  - projects to 2D with UMAP
  - clusters with HDBSCAN (lets natural groups emerge, allows noise)
  - clusters with KMeans (k auto-picked by silhouette score over k=2..8)
  - labels each cluster by its most distinctive TF-IDF terms (c-TF-IDF style)

Embedders:
  - sentence-transformers   semantic, local (all-MiniLM-L6-v2, ~80MB)
  - tfidf                   lexical baseline (TfidfVectorizer, no network)
  - voyage                  Voyage AI API (voyage-3); needs VOYAGE_API_KEY

Outputs to results/<corpus>/definition_clusters/<embedder>/:

  - summary.txt, similarity_distribution.png, similarity_heatmap.png,
    umap_hdbscan.png, umap_kmeans.png, definitions_clustered.jsonl

Run:
    python cluster_definitions.py                              # default: sentence-transformers
    python cluster_definitions.py --embedder tfidf
    python cluster_definitions.py --embedder voyage
    python cluster_definitions.py --all                        # run all three back-to-back
    python cluster_definitions.py --kmin 3 --kmax 10           # tune k-search range
"""

import argparse
import json
import os
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).parent
CORPORA = ("ai-sycophancy",)
FIELD = "Verbatim definition of sycophancy"

EMBEDDERS = ("sentence-transformers", "tfidf", "voyage")
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VOYAGE_MODEL_NAME = "voyage-3"

# Stop tokens that swamp every definition and obscure cluster identity.
DOMAIN_STOPWORDS = {
    "sycophancy", "sycophant", "sycophants",
    "sycophantic", "sycophantically", "sycophancies",
    "model", "models", "ai", "llm", "llms",
    "language", "large", "system", "systems",
    "user", "users", "response", "responses",
    "behavior", "behaviour", "tendency",
}

PALETTE = {
    "primary":   "#243E36",
    "secondary": "#74A57F",
    "accent":    "#ED6A5E",
    "warm":      "#FF8E72",
    "light":     "#FFAF87",
}
PALETTE_ORDER = [
    PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
    PALETTE["warm"], PALETTE["light"],
]

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


def load_definitions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            v = (rec.get(FIELD) or "").strip()
            if not v or v.lower() == "none":
                continue
            rows.append({
                "paper_id": rec.get("paper_id", ""),
                "title": rec.get("title", ""),
                "year": rec.get("year", ""),
                "definition": v,
            })
    return pd.DataFrame(rows)


def embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ST_MODEL_NAME)
    # Normalize so dot product == cosine similarity.
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)


def embed_tfidf(texts: list[str]) -> np.ndarray:
    # Word n-grams up to bigrams. sublinear_tf damps very-frequent terms. L2
    # norm is on by default, so the rows are unit vectors — cosine == dot.
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    X = vec.fit_transform(texts)
    print(f"  TF-IDF vocab: {len(vec.get_feature_names_out())} terms, "
          f"matrix shape {X.shape}")
    return X.toarray().astype(np.float32)


def embed_voyage(texts: list[str], model: str = VOYAGE_MODEL_NAME) -> np.ndarray:
    import voyageai
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("VOYAGE_API_KEY is not set — export it before --embedder voyage")
    vo = voyageai.Client()
    # voyage-3: 1024-dim, 32k-token context. 113 short defs fit in one call;
    # batch defensively in case the corpus grows.
    BATCH = 128
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        result = vo.embed(chunk, model=model, input_type="document",
                          truncation=True)
        out.extend(result.embeddings)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def get_embeddings(method: str, texts: list[str]) -> np.ndarray:
    if method == "sentence-transformers":
        return embed_sentence_transformers(texts)
    if method == "tfidf":
        return embed_tfidf(texts)
    if method == "voyage":
        return embed_voyage(texts)
    raise ValueError(f"unknown embedder: {method!r}")


def best_kmeans(embeddings: np.ndarray, kmin: int, kmax: int) -> tuple[int, np.ndarray, float]:
    """Pick k by silhouette score in [kmin, kmax]."""
    best_k, best_score, best_labels = kmin, -1.0, None
    n = len(embeddings)
    kmax = min(kmax, n - 1)
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(embeddings)
        # silhouette is undefined with only 1 cluster, and unstable with very small clusters.
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        print(f"  k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    return best_k, best_labels, best_score


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    # HDBSCAN expects a distance metric. Cosine distance = 1 - cosine_similarity,
    # but its native cosine support is hit-or-miss; embed-normalized + euclidean
    # is equivalent up to monotonic transform and is reliably supported.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def c_tfidf_terms(definitions: list[str], labels: np.ndarray, top_n: int = 8) -> dict[int, list[str]]:
    """Top distinctive terms per cluster via c-TF-IDF (concatenate per cluster,
    then run TF-IDF with each cluster as one 'document').
    """
    stop = set(TfidfVectorizer(stop_words="english").get_stop_words()) | DOMAIN_STOPWORDS
    by_cluster: dict[int, list[str]] = {}
    for lab, text in zip(labels, definitions):
        by_cluster.setdefault(int(lab), []).append(text)

    clusters = sorted(by_cluster.keys())
    docs = [" ".join(by_cluster[c]) for c in clusters]
    vec = TfidfVectorizer(stop_words=list(stop), ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for i, c in enumerate(clusters):
        row = X[i].toarray().ravel()
        top_idx = np.argsort(-row)[:top_n]
        out[c] = [terms[j] for j in top_idx if row[j] > 0]
    return out


def representative_definitions(df: pd.DataFrame, embeddings: np.ndarray,
                               labels: np.ndarray, n_each: int = 4) -> dict[int, list[int]]:
    """Pick the n_each definitions closest to each cluster's centroid."""
    out: dict[int, list[int]] = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        centroid = embeddings[idx].mean(axis=0)
        sims = embeddings[idx] @ centroid / (np.linalg.norm(centroid) + 1e-12)
        order = idx[np.argsort(-sims)]
        out[int(c)] = order[:n_each].tolist()
    return out


def cluster_color_map(labels: np.ndarray) -> dict[int, tuple]:
    """Blend palette across N clusters; HDBSCAN noise (-1) gets gray."""
    uniq = sorted(set(labels))
    n_real = sum(1 for c in uniq if c >= 0)
    blend = sns.blend_palette(PALETTE_ORDER, n_colors=max(n_real, 2))
    out, i = {}, 0
    for c in uniq:
        if c == -1:
            out[c] = (0.75, 0.75, 0.75, 1.0)
        else:
            out[c] = tuple(blend[i])
            i += 1
    return out


def plot_scatter(coords: np.ndarray, labels: np.ndarray, title: str, out: Path,
                 term_labels: dict[int, list[str]] | None = None) -> None:
    cmap = cluster_color_map(labels)
    fig, ax = plt.subplots(figsize=(12, 8))
    for c in sorted(set(labels)):
        mask = labels == c
        name = "noise" if c == -1 else f"cluster {c}"
        if term_labels and c != -1 and term_labels.get(c):
            name += f" ({', '.join(term_labels[c][:3])})"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=cmap[c], label=name, s=80, alpha=0.85,
                   edgecolor="white", linewidth=0.8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=14)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_similarity_histogram(sim: np.ndarray, out: Path) -> None:
    iu = np.triu_indices_from(sim, k=1)
    vals = sim[iu]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.hist(vals, bins=40, color=PALETTE["primary"], edgecolor="white")
    median = float(np.median(vals))
    ax.axvline(median, color=PALETTE["accent"], linestyle="--", linewidth=2,
               label=f"median = {median:.2f}")
    ax.set_xlabel("Pairwise cosine similarity")
    ax.set_ylabel("Pairs")
    ax.set_title("Divergence: distribution of pairwise definition similarity")
    ax.legend(frameon=False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_similarity_heatmap(sim: np.ndarray, labels: np.ndarray, out: Path) -> None:
    # Re-sort so cluster blocks line up along the diagonal.
    order = np.argsort(labels, kind="stable")
    sorted_sim = sim[order][:, order]
    cmap = sns.light_palette(PALETTE["primary"], as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(sorted_sim, cmap=cmap, ax=ax, square=True,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "cosine similarity"})
    ax.set_title("Pairwise similarity (sorted by KMeans cluster)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def write_summary(out: Path, df: pd.DataFrame, sim: np.ndarray,
                  km_k: int, km_labels: np.ndarray, km_score: float,
                  hd_labels: np.ndarray,
                  km_terms: dict[int, list[str]], hd_terms: dict[int, list[str]],
                  km_reps: dict[int, list[int]], hd_reps: dict[int, list[int]]) -> None:
    lines: list[str] = []
    iu = np.triu_indices_from(sim, k=1)
    vals = sim[iu]
    lines.append(f"Definitions clustered: {len(df)}")
    lines.append(f"Unique definitions:    {df['definition'].nunique()}")
    lines.append("")
    lines.append("Pairwise cosine similarity (all unique pairs):")
    lines.append(f"  min:    {vals.min():.3f}")
    lines.append(f"  median: {np.median(vals):.3f}")
    lines.append(f"  mean:   {vals.mean():.3f}")
    lines.append(f"  max:    {vals.max():.3f}")
    lines.append(f"  std:    {vals.std():.3f}")
    lines.append("")

    def render(header: str, labels: np.ndarray, terms: dict[int, list[str]],
               reps: dict[int, list[int]]) -> None:
        lines.append(header)
        for c in sorted(set(labels)):
            mask = labels == c
            tag = "NOISE" if c == -1 else f"Cluster {c}"
            lines.append(f"  {tag} (n={int(mask.sum())})")
            if terms.get(c):
                lines.append(f"    top terms: {', '.join(terms[c])}")
            for i in reps.get(int(c), []):
                row = df.iloc[i]
                snippet = row["definition"]
                if len(snippet) > 280:
                    snippet = snippet[:277] + "..."
                lines.append(f"    - [{row['year']}] {snippet}")
            lines.append("")

    render(f"KMeans clustering  (best k={km_k}, silhouette={km_score:.3f}):",
           km_labels, km_terms, km_reps)
    render("HDBSCAN clustering (natural clusters, -1 = noise):",
           hd_labels, hd_terms, hd_reps)

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--corpus", choices=CORPORA, default="ai-sycophancy")
    p.add_argument("--input", type=Path, default=None,
                   help="explicit annotations jsonl (default: data/<corpus>/annotations.jsonl)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="base output dir (default: results/<corpus>/definition_clusters/); "
                        "each embedder gets its own sub-directory")
    p.add_argument("--embedder", choices=EMBEDDERS, default="sentence-transformers",
                   help="embedding method (default: sentence-transformers)")
    p.add_argument("--all", action="store_true",
                   help="run every embedder in EMBEDDERS, one after the other")
    p.add_argument("--kmin", type=int, default=2, help="min k to try for KMeans (default 2)")
    p.add_argument("--kmax", type=int, default=8, help="max k to try for KMeans (default 8)")
    p.add_argument("--hdbscan-min", type=int, default=5,
                   help="HDBSCAN min_cluster_size (default 5)")
    p.add_argument("--seed", type=int, default=0, help="UMAP random seed (default 0)")
    return p.parse_args()


def run_pipeline(method: str, df: pd.DataFrame, base_outdir: Path,
                 args: argparse.Namespace) -> None:
    outdir = base_outdir / method
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Embedder: {method}  ->  {outdir} ===")

    print("Embedding...")
    embeddings = get_embeddings(method, df["definition"].tolist())
    print(f"  embeddings shape: {embeddings.shape}")

    print("Computing pairwise similarity...")
    sim = cosine_similarity(embeddings)

    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=min(15, len(df) - 1), min_dist=0.1,
                        metric="cosine", random_state=args.seed)
    coords = reducer.fit_transform(embeddings)

    print(f"KMeans search k={args.kmin}..{args.kmax}:")
    km_k, km_labels, km_score = best_kmeans(embeddings, args.kmin, args.kmax)
    print(f"  -> best k={km_k} (silhouette={km_score:.3f})")

    print(f"HDBSCAN (min_cluster_size={args.hdbscan_min})...")
    hd_labels = cluster_hdbscan(embeddings, args.hdbscan_min)
    n_hd_clusters = len(set(hd_labels) - {-1})
    n_noise = int((hd_labels == -1).sum())
    print(f"  -> {n_hd_clusters} cluster(s), {n_noise} noise point(s)")

    print("Labeling clusters with c-TF-IDF terms...")
    km_terms = c_tfidf_terms(df["definition"].tolist(), km_labels)
    hd_terms = c_tfidf_terms(df["definition"].tolist(), hd_labels)

    km_reps = representative_definitions(df, embeddings, km_labels)
    hd_reps = representative_definitions(df, embeddings, hd_labels)

    print("Plotting...")
    plot_similarity_histogram(sim, outdir / "similarity_distribution.png")
    plot_similarity_heatmap(sim, km_labels, outdir / "similarity_heatmap.png")
    plot_scatter(coords, km_labels,
                 f"Definition clusters — {method} (KMeans, k={km_k})",
                 outdir / "umap_kmeans.png", term_labels=km_terms)
    plot_scatter(coords, hd_labels,
                 f"Definition clusters — {method} (HDBSCAN, {n_hd_clusters} clusters + noise)",
                 outdir / "umap_hdbscan.png", term_labels=hd_terms)

    write_summary(outdir / "summary.txt", df, sim, km_k, km_labels, km_score,
                  hd_labels, km_terms, hd_terms, km_reps, hd_reps)

    out_jsonl = outdir / "definitions_clustered.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            rec = {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "year": row["year"],
                "definition": row["definition"],
                "embedder": method,
                "kmeans_cluster": int(km_labels[i]),
                "hdbscan_cluster": int(hd_labels[i]),
                "umap_x": float(coords[i, 0]),
                "umap_y": float(coords[i, 1]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_jsonl}")


def main() -> None:
    args = parse_args()
    input_path = args.input or HERE / "data" / args.corpus / "annotations.jsonl"
    base_outdir = args.outdir or HERE / "results" / args.corpus / "definition_clusters"
    base_outdir.mkdir(parents=True, exist_ok=True)

    df = load_definitions(input_path)
    print(f"Loaded {len(df)} non-empty verbatim definitions from {input_path}")
    n_dup = len(df) - df["definition"].nunique()
    if n_dup:
        print(f"  ({n_dup} exact duplicates retained)")

    methods = list(EMBEDDERS) if args.all else [args.embedder]
    for method in methods:
        run_pipeline(method, df, base_outdir, args)


if __name__ == "__main__":
    main()
