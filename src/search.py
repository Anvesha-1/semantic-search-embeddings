"""Member 3: semantic search over the article embeddings using FAISS."""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

try:
    from src.embedder import embed_texts   # when imported from the project root (e.g. the web app)
except ImportError:
    from embedder import embed_texts       # when run as: python src/search.py

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.csv"
EMB_PATH = ROOT / "data" / "processed" / "corpus_embeddings.npy"


class SemanticSearcher:
    """Finds the articles whose meaning is closest to the query.

    search() returns the same format as KeywordSearcher.search():
    a list of dicts with doc_id, text, category and score.
    """

    def __init__(self):
        self.df = pd.read_csv(CORPUS_PATH)
        embeddings = np.ascontiguousarray(np.load(EMB_PATH), dtype=np.float32)

        # Row i of the embeddings must match doc_id i in corpus.csv
        assert len(self.df) == embeddings.shape[0], "Corpus and embeddings differ in size!"

        # IndexFlatIP = exact search by inner product. The vectors are normalized,
        # so inner product equals cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        query_vec = np.ascontiguousarray(embed_texts([query]), dtype=np.float32)
        scores, ids = self.index.search(query_vec, top_k)

        results = []
        for score, row in zip(scores[0], ids[0]):
            if row == -1:   # FAISS returns -1 when there are fewer than top_k results
                continue
            article = self.df.iloc[int(row)]
            results.append({
                "doc_id": int(article["doc_id"]),
                "text": article["text"],
                "category": article["category"],
                "score": round(float(score), 4),
            })
        return results


if __name__ == "__main__":
    searcher = SemanticSearcher()
    for q in ["oil prices rise", "cost of fuel goes up", "team lifts the trophy after a long wait"]:
        print(f"\nQuery: {q}")
        for r in searcher.search(q, top_k=3):
            print(f"  {r['score']:.3f} [{r['category']}] {r['text'][:80]}")