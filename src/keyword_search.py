"""
Member 1 - Keyword search baseline (TF-IDF)
A simple keyword-based search engine. The team compares semantic search
against it. It has the SAME interface as SemanticSearcher (Member 3),
so the app and the evaluation can switch between the two.

Run a quick demo:   python src/keyword_search.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

CORPUS_PATH = Path("data/processed/corpus.csv")


class KeywordSearcher:
    def __init__(self, corpus_path=CORPUS_PATH):
        self.df = pd.read_csv(corpus_path)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",   # ignore words like "the", "is", "and"
            ngram_range=(1, 2),     # use single words and word pairs
            min_df=2,               # ignore words that appear in only 1 article
            sublinear_tf=True,      # dampen very frequent words
        )
        # one TF-IDF row per article (rows are already length-normalised)
        self.matrix = self.vectorizer.fit_transform(self.df["text"])

    def search(self, query, top_k=5):
        """Return the top_k articles for a query, best first."""
        query_vec = self.vectorizer.transform([query])
        scores = linear_kernel(query_vec, self.matrix).ravel()  # = cosine similarity
        best = np.argsort(-scores)[:top_k]
        return [
            {
                "doc_id": int(self.df.iloc[i]["doc_id"]),
                "text": self.df.iloc[i]["text"],
                "category": self.df.iloc[i]["category"],
                "score": round(float(scores[i]), 4),
            }
            for i in best
        ]


if __name__ == "__main__":
    searcher = KeywordSearcher()
    print(f"Vocabulary size: {len(searcher.vectorizer.vocabulary_)} terms\n")
    for q in ["oil prices rise", "cost of fuel goes up"]:
        print(f"Query: {q}")
        for r in searcher.search(q, top_k=3):
            print(f"  [{r['score']:.3f}] ({r['category']}) {r['text'][:90]}")
        print()