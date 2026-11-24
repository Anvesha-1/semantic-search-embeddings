# Chapter 4: Web Application

*Written by Member 4 (Mansi).*

## 4.1 Goal

The search engines from Chapters 1 and 3 work from the command line. The goal of this part is to put them behind a simple web page, so that anyone can type a query, pick a search method and read the results without writing any code.

## 4.2 Technology

The app is built with **Streamlit**, a Python library that turns a Python script into an interactive web page. It needs no HTML, CSS or JavaScript. The whole app is one file, `app.py`, which imports the two search classes built by the team:

| Class | File | Built in |
|---|---|---|
| `SemanticSearcher` | `src/search.py` | Chapter 3 |
| `KeywordSearcher` | `src/keyword_search.py` | Chapter 1 |

Both classes have the same `search(query, top_k)` method and return results in the same format (`doc_id`, `text`, `category`, `score`). This is why the app can switch between them with a single drop-down menu.

## 4.3 How to run it

From the project folder, with the virtual environment active:

```
streamlit run app.py
```

Streamlit starts a local web server and opens the app in the browser at `http://localhost:8501`.

## 4.4 Features

**Sidebar (search settings)**

- **Search method:** a drop-down to choose *Semantic Search* or *Keyword Search*.
- **Number of results:** a slider from 1 to 20 (default 5).

**Main page**

- A text box for the query, with an example shown as a placeholder.
- A **Search Articles** button.
- The results, each in its own bordered box showing:
  - the rank and the article number (`doc_id`)
  - the article's **category**
  - the **similarity score**, with 4 decimal places
  - the full article text

## 4.5 How it works

1. The user types a query and clicks **Search Articles**.
2. The app checks which method is selected in the sidebar and loads that search engine.
3. It calls `searcher.search(query, top_k)` and displays the returned list.

**Caching.** Loading a search engine is the slow part. The semantic engine has to load the 5,000 embeddings, build the FAISS index and load the embedding model. The keyword engine has to build its TF-IDF matrix. Both loading functions use Streamlit's `@st.cache_resource` decorator, so each engine is built **once**, the first time it is used. After that, every search reuses it and returns almost instantly.

**Error handling.**

- An empty query shows a warning ("Please enter a search query.") instead of searching.
- If the data or embedding files are missing, the app shows a clear error asking the user to generate the project data first.
- Any other error is shown as a message on the page rather than crashing the app.

## 4.6 Example: the same query in both modes

The query **"cost of fuel goes up"** shows the difference between the two methods. In *Keyword Search* mode, the top 3 results are:

| Rank | Category | Score | Article (start) |
|---|---|---|---|
| 1 | Sci/Tech | 0.247 | Battered Florida Counts the Cost After Frances... |
| 2 | Sports | 0.179 | US goes distance in softball... |
| 3 | Business | 0.174 | Turbulent times for US airline companies... skyrocketing fuel prices... |

The first two results only matched single words ("cost" and "goes") and have nothing to do with fuel prices. In *Semantic Search* mode, the same query is matched by meaning. Chapter 3 measured this query at 5 out of 5 results in the expected Business category for semantic search, against 1 out of 5 for keyword search.

This side-by-side use of the two modes is the easiest way to demonstrate the project during the presentation.

## 4.7 Testing

The app was tested from a fresh download of the GitHub repository:

- `pip install -r requirements.txt` installed Streamlit and all other libraries without errors.
- The app loaded, and a search returned results with no errors in both *Semantic Search* and *Keyword Search* modes.
- Each result showed its rank, article number, category, score and text.

## 4.8 Limitations and future work

- **First search is slower.** The first semantic search takes a few seconds while the embedding model loads; later searches are fast because of caching.
- **Runs locally.** The app runs on the user's own computer. It could be deployed online, for example on Streamlit Community Cloud, so that others can use it from a link.
- **Possible improvements:**
  - a category filter, to search within one category only
  - a "Compare" mode showing keyword and semantic results side by side on one screen
  - highlighting the words in each result that match the query

## 4.9 Summary

- `app.py` gives the project a simple web interface built with Streamlit.
- Users choose the search method and number of results, type a query and read the results with their category and similarity score.
- Both search engines are cached, so searches after the first are fast.
- Running the same query in both modes shows the main result of the project: semantic search finds articles by meaning, even when the words are different.

## Files for this chapter

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `src/search.py` | Semantic search engine used by the app |
| `src/keyword_search.py` | Keyword search engine used by the app |
