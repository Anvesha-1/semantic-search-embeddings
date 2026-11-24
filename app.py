
import streamlit as st

from src.search import SemanticSearcher
from src.keyword_search import KeywordSearcher


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Semantic News Search",
    page_icon="🔎",
    layout="wide"
)


# ---------------- LOAD SEARCH ENGINES ----------------

@st.cache_resource
def load_semantic_searcher():
    return SemanticSearcher()


@st.cache_resource
def load_keyword_searcher():
    return KeywordSearcher()


# ---------------- APPLICATION HEADER ----------------

st.title("🔎 Semantic News Search Engine")

st.markdown(
    """
    Search news articles using **Artificial Intelligence
    and Natural Language Processing**.

    Find articles based on their meaning rather than
    simply matching keywords.
    """
)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.header("Search Settings")

search_type = st.sidebar.selectbox(
    "Select Search Method",
    ["Semantic Search", "Keyword Search"]
)

top_k = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=20,
    value=5
)


# ---------------- SEARCH INPUT ----------------

st.subheader("Enter Your Search Query")

query = st.text_input(
    "What news are you looking for?",
    placeholder="Example: Rising oil prices and global economy"
)

search_button = st.button(
    "🔍 Search Articles",
    type="primary"
)


# ---------------- SEARCH EXECUTION ----------------

if search_button:

    if not query.strip():
        st.warning("Please enter a search query.")

    else:
        try:
            with st.spinner("Searching relevant articles..."):

                if search_type == "Semantic Search":
                    searcher = load_semantic_searcher()
                else:
                    searcher = load_keyword_searcher()

                results = searcher.search(
                    query,
                    top_k=top_k
                )

            st.divider()

            st.subheader("Search Results")

            st.write(
                f"Found {len(results)} relevant articles "
                f"using {search_type}."
            )

            if not results:
                st.info("No matching articles found.")

            for i, article in enumerate(results, start=1):

                with st.container(border=True):

                    st.markdown(
                        f"### {i}. Article #{article['doc_id']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"**Category:** {article['category']}"
                        )

                    with col2:
                        st.markdown(
                            f"**Similarity Score:** "
                            f"{article['score']:.4f}"
                        )

                    st.write(article["text"])

        except FileNotFoundError as e:
            st.error(
                "Required dataset or embedding files are missing. "
                "Please generate the project data first."
            )
            st.code(str(e))

        except Exception as e:
            st.error(f"An error occurred: {e}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Semantic Search with Embeddings | "
    "AG News Dataset | FAISS + Sentence Transformers + TF-IDF"
)
