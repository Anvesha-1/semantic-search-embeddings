"""Member 3: evaluate semantic search vs keyword search with Precision@5."""

from pathlib import Path

import pandas as pd

from search import SemanticSearcher
from keyword_search import KeywordSearcher

ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = ROOT / "data" / "eval" / "test_queries.csv"
K = 5

queries = pd.read_csv(QUERIES_PATH)
semantic = SemanticSearcher()
keyword = KeywordSearcher()


def precision_at_k(searcher, query, expected_category):
    """Fraction of the top-K results that are in the expected category."""
    results = searcher.search(query, top_k=K)
    if not results:
        return 0.0
    hits = sum(1 for r in results if r["category"] == expected_category)
    return hits / K


rows = []
for _, q in queries.iterrows():
    rows.append({
        "query_id": q["query_id"],
        "query": q["query"],
        "type": q["type"],
        "expected": q["expected_category"],
        "semantic": precision_at_k(semantic, q["query"], q["expected_category"]),
        "keyword": precision_at_k(keyword, q["query"], q["expected_category"]),
    })
per_query = pd.DataFrame(rows)

# Summary table (in percent)
summary = {"Overall": per_query[["semantic", "keyword"]].mean()}
for t in ["direct", "paraphrase"]:
    summary[t] = per_query[per_query["type"] == t][["semantic", "keyword"]].mean()
for c in sorted(per_query["expected"].unique()):
    summary[c] = per_query[per_query["expected"] == c][["semantic", "keyword"]].mean()
table = (pd.DataFrame(summary).T * 100).round(1)
table.columns = ["Semantic search", "Keyword search"]

print(f"\nPrecision@{K} (%), 40 test queries:\n")
print(table.to_string())

# Where does each method fail?
print("\nSemantic search got fewer than 5/5 on these queries:")
weak = per_query[per_query["semantic"] < 1.0]
for _, r in weak.iterrows():
    print(f"  [{r['type']}] '{r['query']}' (expected {r['expected']}): {r['semantic']*5:.0f}/5")

print("\nKeyword search scored 0/5 on these queries:")
zero = per_query[per_query["keyword"] == 0.0]
for _, r in zero.iterrows():
    print(f"  [{r['type']}] '{r['query']}' (expected {r['expected']})")

(ROOT / "results").mkdir(exist_ok=True)
table.to_csv(ROOT / "results" / "evaluation_summary.csv")
per_query.to_csv(ROOT / "results" / "evaluation_per_query.csv", index=False)
print("\nSaved results/evaluation_summary.csv and results/evaluation_per_query.csv")