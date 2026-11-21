# Semantic Search with Embeddings

Deep Learning mini project. Search news articles by meaning
using sentence embeddings instead of keyword matching.

## Team
| Member | Part |
|---|---|
| <Anvesha> | Data preparation, data analysis, test queries, keyword search baseline, final testing |
| <Name 2> | Embedding generation, model comparison, embedding visualization |
| <Name 3> | Semantic search engine (FAISS) and evaluation |
| <Name 4> | Streamlit web app, README, final report and slides |

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
TODO (FILL YOURSELF)

## Search and Evaluation (Member 3)
TODO (FILL YOURSELF)

## Web App (Member 4)
TODO (FILL YOURSELF)