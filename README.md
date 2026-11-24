# Semantic Search with Embeddings

Deep Learning mini project. Search news articles by meaning
using sentence embeddings instead of keyword matching.

## Team
| Member | Part |
|---|---|
| Anvesha | Data preparation, data analysis, test queries, keyword search baseline, final testing |
| Mahima | Embedding generation, model comparison, embedding visualization |
| Harsh | Semantic search engine (FAISS) and evaluation |
| Mansi | Streamlit web app, README, final report and slides |

## Setup
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Data, Test Set and Keyword Baseline (Member 1)

### Dataset
- Source: AG News (Hugging Face: fancyzhx/ag_news), 120,000 news articles in 4 categories.
- Balanced sample of 5,000 articles (1,250 per category), random seed 42.
- Cleaning: fixed HTML codes like #39; and &lt;b&gt;, removed tags and extra spaces,
  dropped texts under 8 words and exact duplicates.
- Output: data/processed/corpus.csv (doc_id, text, label, category)
- Recreate: python src/data_prep.py

### Data analysis
- python src/explore_data.py  ->  charts in results/figures/, stats in results/data_stats.md

### Test queries
- data/eval/test_queries.csv: 40 queries, 10 per category.
- 20 direct queries use words found in the articles; 20 paraphrase queries
  say the same thing in different words, to test search by meaning.

### Keyword search baseline
- src/keyword_search.py: TF-IDF + cosine similarity (scikit-learn).
- KeywordSearcher().search(query, top_k=5) returns the same format as SemanticSearcher.
- Demo: python src/keyword_search.py

## Embeddings (Member 2)
- Model: all-MiniLM-L6-v2 (sentence-transformers), 384-dimensional vectors, normalized so cosine similarity = inner product.
- src/embedder.py: embed_texts(texts) returns a float32 array of shape (n, 384). Always use this function to embed queries.
- Create corpus embeddings: python src/embed_corpus.py -> data/processed/corpus_embeddings.npy, shape (5000, 384). Row i matches doc_id i in corpus.csv.
- Meaning check: python src/test_embedder.py. "cost of fuel goes up" vs "oil prices rise" scores 0.504, versus 0.022 for "football coach fired".
- Model comparison: python src/compare_models.py. all-MiniLM-L6-v2 scores 93.5% Precision@5 vs 91.5% for all-mpnet-base-v2, and is about 5.6 times faster. Results   in results/model_comparison.csv.
- 2D plot: python src/visualize_embeddings.py -> results/figures/embeddings_2d.png

## Search and Evaluation (Member 3)
- src/search.py: SemanticSearcher().search(query, top_k=5) uses FAISS (IndexFlatIP, exact inner-product search on normalized embeddings). Returns a list of dicts     with doc_id, text, category and score, the same format as KeywordSearcher.
- Evaluate: python src/evaluate.py runs the 40 test queries on both searches -> results/evaluation_summary.csv and results/evaluation_per_query.csv.
- Result: Precision@5 is 93.5% for semantic search vs 84.0% for keyword search. On paraphrase queries it is 92.0% vs 72.0%. On direct queries the two are about       equal (95.0% vs 96.0%).
## Web App (Member 4)
- app.py: Streamlit web app for searching the news articles.
- Start it from the project folder: streamlit run app.py  (opens in the browser at http://localhost:8501)
- Sidebar: choose Semantic Search or Keyword Search, and set the number of results (1-20).
- Type a query and click "Search Articles" to see each result's category and similarity score.
- Tip: try a paraphrase such as "cost of fuel goes up" in both modes to see the difference.
- The first semantic search takes a few seconds while the embedding model loads; later searches are fast.

## Report
- Full project report: docs/report.md (chapters 1-4 are also in docs/report/)
