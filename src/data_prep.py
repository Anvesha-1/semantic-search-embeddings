"""
Member 1 - Data preparation
Loads the AG News dataset, cleans it, takes a balanced sample
and saves it as data/processed/corpus.csv for the rest of the team.

Run from the project folder:   python src/data_prep.py
"""
import html
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------- settings ----------------
SAMPLES_PER_CATEGORY = 1250        # 4 categories x 1250 = 5000 articles
MIN_WORDS = 8                      # drop texts shorter than this
RANDOM_SEED = 42                   # same seed = same sample every time
CATEGORIES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
OUTPUT_PATH = Path("data/processed/corpus.csv")


def load_raw_data():
    """Download AG News (train split) and return it as a pandas DataFrame."""
    dataset = load_dataset("fancyzhx/ag_news", split="train")
    return dataset.to_pandas()          # columns: text, label


def clean_text(text):
    """Fix the messy characters that appear in AG News."""
    text = re.sub(r"&?#39;", "'", text)         # #39;  -> '
    text = re.sub(r"&?quot;", '"', text)        # quot; -> "
    text = html.unescape(text)                   # &amp; -> &, &lt; -> <
    text = re.sub(r"<[^>]+>", " ", text)         # remove HTML tags like <b>
    text = text.replace("\\", " ")               # stray backslashes
    text = re.sub(r"\s+", " ", text)             # many spaces -> one space
    return text.strip()


def prepare_corpus(df):
    """Clean, filter, de-duplicate and sample the data."""
    print(f"Raw rows: {len(df)}")

    df["text"] = df["text"].apply(clean_text)

    # remove very short texts
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    # remove exact duplicates
    df = df.drop_duplicates(subset="text")
    print(f"After cleaning: {len(df)}")

    # balanced sample: same number of articles from each category
    df = (
        df.groupby("label", group_keys=False)
        .sample(n=SAMPLES_PER_CATEGORY, random_state=RANDOM_SEED)
        .sample(frac=1, random_state=RANDOM_SEED)    # shuffle rows
        .reset_index(drop=True)
    )

    # final columns (this format is fixed - the team depends on it)
    df["doc_id"] = range(len(df))
    df["category"] = df["label"].map(CATEGORIES)
    return df[["doc_id", "text", "label", "category"]]


def main():
    raw = load_raw_data()
    corpus = prepare_corpus(raw)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(corpus)} rows to {OUTPUT_PATH}")
    print("\nArticles per category:")
    print(corpus["category"].value_counts())
    print("\nExample row:")
    print(corpus.iloc[0].to_dict())


if __name__ == "__main__":
    main()