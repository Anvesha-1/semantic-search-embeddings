"""Member 2: compare two embedding models with Precision@5 on the 40 test queries."""

import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from embedder import embed_texts

CORPUS_PATH = "data/processed/corpus.csv"
QUERIES_PATH = "data/eval/test_queries.csv"
EMB_PATH = "data/processed/corpus_embeddings.npy"
MODEL_1 = "all-MiniLM-L6-v2"
MODEL_1_SECONDS = 66.6  # from your own embed_corpus.py run
MODEL_2 = "all-mpnet-base-v2"
K = 5

corpus = pd.read_csv(CORPUS_PATH)
queries = pd.read_csv(QUERIES_PATH)
doc_categories = corpus["category"].to_numpy(dtype=str)
expected = queries["expected_category"].to_numpy(dtype=str)
types = queries["type"].to_numpy(dtype=str)
query_texts = queries["query"].tolist()

assert set(expected) <= set(doc_categories), "Category names do not match!"


def precision_at_k(query_vecs, doc_vecs):
    """For each query: fraction of the top-K articles in the expected category."""
    scores = query_vecs @ doc_vecs.T          # cosine similarity (vectors are normalized)
    top = np.argsort(-scores, axis=1)[:, :K]  # indices of the K best articles
    return (doc_categories[top] == expected[:, None]).mean(axis=1)


def summarize(hits):
    out = {"Overall": hits.mean()}
    for t in ["direct", "paraphrase"]:
        out[t] = hits[types == t].mean()
    for c in sorted(set(expected)):
        out[c] = hits[expected == c].mean()
    return out


# Model 1: reuse the embeddings you already saved
docs_1 = np.load(EMB_PATH)
hits_1 = precision_at_k(embed_texts(query_texts), docs_1)

# Model 2: embed the whole corpus again with the bigger model
print(f"Loading {MODEL_2} (first run downloads about 420 MB)...")
model_2 = SentenceTransformer(MODEL_2)
start = time.time()
docs_2 = model_2.encode(
    corpus["text"].tolist(), batch_size=64, normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=True,
)
seconds_2 = time.time() - start
queries_2 = model_2.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
hits_2 = precision_at_k(queries_2, docs_2)

# Summary table (in percent)
table = pd.DataFrame({MODEL_1: summarize(hits_1), MODEL_2: summarize(hits_2)})
table = (table * 100).round(1)
table.loc["Vector size"] = [docs_1.shape[1], docs_2.shape[1]]
table.loc["Corpus embedding time (s)"] = [MODEL_1_SECONDS, round(seconds_2, 1)]

print("\nPrecision@5 (%), by model:\n")
print(table.to_string())

table.to_csv("results/model_comparison.csv")
pd.DataFrame({
    "query": query_texts, "type": queries["type"], "expected": expected,
    MODEL_1: hits_1, MODEL_2: hits_2,
}).to_csv("results/model_comparison_per_query.csv", index=False)
print("\nSaved results/model_comparison.csv and results/model_comparison_per_query.csv")