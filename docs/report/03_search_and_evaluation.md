# Chapter 3: Semantic Search and Evaluation

*Written by Member 3.*

## 3.1 Goal

The goal of this part is to build a search engine that finds articles by **meaning**, and to measure how much better it is than ordinary keyword search.

## 3.2 How the search works

1. **Offline step (done once).**Every article was turned into a 384-number vector in Chapter 2. These vectors are stored in `data/processed/corpus_embeddings.npy`.
2. **At search time.** The user's query is turned into a vector using the same function, `embed_texts()`, and the same model. Using the same model matters, because vectors from different models cannot be compared.
3. **Finding the closest articles.** The query vector is compared with all 5,000 article vectors and the 5 closest are returned.

### Cosine similarity

Cosine similarity measures how close two vectors point in the same direction. A score near 1 means very similar meaning, and a score near 0 means unrelated. All vectors in this project are normalized to length 1, so cosine similarity equals the **inner product** (the sum of the products of the matching numbers).

### FAISS and IndexFlatIP

FAISS is a library for fast vector search. We use `IndexFlatIP`:

- **Flat** means it compares the query with every vector, so the result is exact. For 5,000 articles this is instant, and there is no need for an approximate index.
- **IP** means inner product. Because the vectors are normalized, the inner product ranks the articles in the same order as cosine similarity.

### The SemanticSearcher class

`src/search.py` contains `SemanticSearcher`. Its `search(query, top_k=5)` method returns a list of results with `doc_id`, `text`, `category` and `score`. This is the same format as the keyword search from Chapter 1, so the web app can switch between the two by changing one line.

Row `i` of the embeddings file matches `doc_id` `i` in `corpus.csv`. The code checks that both have the same length when it starts.

## 3.3 How we measure quality

We use the 40 test queries from Chapter 1: 10 per category, half **direct** (using words found in the articles) and half **paraphrase** (saying the same thing in different words).

**Precision@5.** For each query we take the top 5 results and count how many are in the expected category. A query scores 5/5 = 100% if all 5 results are in the right category. We then average over the queries.

This measure checks the **category** of the results, not whether each article truly answers the query. It is a rough but fair way to compare the two methods on the same queries.

## 3.4 Results

| Precision@5 (%) | Semantic search | Keyword search |
|---|---|---|
| Overall | **93.5** | 84.0 |
| Direct queries | 95.0 | **96.0** |
| Paraphrase queries | **92.0** | 72.0 |
| Business | **90.0** | 78.0 |
| Sci/Tech | **94.0** | 90.0 |
| Sports | **94.0** | 80.0 |
| World | **96.0** | 88.0 |

The full results are in `results/evaluation_summary.csv` and `results/evaluation_per_query.csv`.

## 3.5 Discussion

1. **Semantic search is better overall**, 93.5% against 84.0%.
2. **The big difference is on paraphrase queries.** When the query uses different words from the article, semantic search keeps 92% while keyword search drops to 72%, a gap of 20 points. Keyword search can only match words, so a query like "cost of fuel goes up" cannot find an article that says "oil prices rise". Semantic search compares meaning, so it can.
3. **On direct queries the two methods are about equal.** Keyword search scores 96% against 95%. When the query contains the exact words of the article, word matching works very well. Semantic search does not lose much here, but it does not win either.
4. **Semantic search wins in every category.** The gap is largest in Business (12 points) and Sports (14 points). It is smallest in Sci/Tech (4 points), where the technical vocabulary is distinctive and keywords already work well.
5. **Sanity check.** Our semantic scores equal the scores that Member 2 measured separately for the same model (93.5% overall, 95.0% direct, 92.0% paraphrase). This shows that the FAISS search and the direct matrix calculation give the same ranking, as expected for an exact index.

## 3.6 Limitations

- There are only 40 queries, so one query changes a category score by 10 points and the overall score by 2.5 points. Small differences, such as 95% against 96% on direct queries, should not be over-read.
- Precision@5 checks the category, not true relevance. An article in the right category may still not answer the query, and a result in a neighbouring category (for example Business instead of Sci/Tech) may actually be useful.
- The articles are short (about 38 words on average), and the corpus has only 5,000 articles. Results on longer documents or a bigger collection could be different.
- An exact flat index is fine at this size. For millions of articles, an approximate index would be needed.

## 3.7 Summary

- We built `SemanticSearcher`, an exact FAISS search over 5,000 normalized embeddings.
- Semantic search reached **93.5%** Precision@5, against **84.0%** for keyword search.
- The advantage comes from paraphrase queries: **92%** against **72%**.
- On direct queries, keyword search is as good as semantic search.

## Files for this chapter

| File | Purpose |
|---|---|
| `src/search.py` | `SemanticSearcher` class (FAISS) |
| `src/evaluate.py` | Runs the 40 test queries on both methods |
| `results/evaluation_summary.csv` | Precision@5 summary table |
| `results/evaluation_per_query.csv` | Result for every query |
