"""
Member 1 - Exploratory data analysis
Makes charts and a statistics file from corpus.csv for the report.

Run:   python src/explore_data.py
Output: results/figures/*.png  and  results/data_stats.md
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CORPUS_PATH = Path("data/processed/corpus.csv")
FIG_DIR = Path("results/figures")
STATS_PATH = Path("results/data_stats.md")


def top_words(texts, n=10):
    """Most common meaningful words in a list of texts."""
    counter = Counter()
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        counter.update(w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2)
    return [word for word, _ in counter.most_common(n)]


def main():
    df = pd.read_csv(CORPUS_PATH)
    df["n_words"] = df["text"].str.split().str.len()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: articles per category
    counts = df["category"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="steelblue")
    ax.set_title("Articles per category")
    ax.set_ylabel("Number of articles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "category_counts.png", dpi=150)
    plt.close(fig)

    # Chart 2: distribution of article length
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["n_words"], bins=40, color="steelblue")
    ax.set_title("Article length")
    ax.set_xlabel("Words per article")
    ax.set_ylabel("Number of articles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_histogram.png", dpi=150)
    plt.close(fig)

    # Chart 3: article length by category
    cats = sorted(df["category"].unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([df.loc[df["category"] == c, "n_words"] for c in cats])
    ax.set_xticks(range(1, len(cats) + 1), cats)
    ax.set_title("Article length by category")
    ax.set_ylabel("Words per article")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_by_category.png", dpi=150)
    plt.close(fig)

    # Statistics file for the report
    lines = [
        "# Dataset statistics",
        "",
        f"- Total articles: {len(df)}",
        f"- Average length: {df['n_words'].mean():.1f} words",
        f"- Shortest: {df['n_words'].min()} words, longest: {df['n_words'].max()} words",
        "",
        "| Category | Articles | Avg words | Top words |",
        "|---|---|---|---|",
    ]
    for c in cats:
        part = df[df["category"] == c]
        lines.append(
            f"| {c} | {len(part)} | {part['n_words'].mean():.1f} | {', '.join(top_words(part['text'], 8))} |"
        )
    STATS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nCharts saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()