import os
import warnings

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bible Verse Recommender and Visualization", layout="wide")

# ── Inline stopword list — no nltk needed ─────────────────────────────────────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "that", "this", "these",
    "those", "it", "its", "he", "she", "they", "we", "you", "i", "me",
    "him", "her", "them", "us", "my", "his", "our", "your", "their", "not",
    "no", "nor", "so", "yet", "both", "either", "each", "all", "any",
    "more", "most", "other", "such", "than", "then", "when", "where",
    "which", "who", "whom", "what", "how", "if", "as", "up", "out", "into",
    "about", "after", "before", "over", "under", "again", "there", "here",
    "now", "just", "also", "very", "own", "same", "only", "because",
}

# ── Book name maps ────────────────────────────────────────────────────────────
ALL_BOOK_NAMES = {
    1: "Genesis", 2: "Exodus", 3: "Leviticus", 4: "Numbers", 5: "Deuteronomy",
    6: "Joshua", 7: "Judges", 8: "Ruth", 9: "1 Samuel", 10: "2 Samuel",
    11: "1 Kings", 12: "2 Kings", 13: "1 Chronicles", 14: "2 Chronicles",
    15: "Ezra", 16: "Nehemiah", 17: "Esther", 18: "Job", 19: "Psalms",
    20: "Proverbs", 21: "Ecclesiastes", 22: "Song of Solomon", 23: "Isaiah",
    24: "Jeremiah", 25: "Lamentations", 26: "Ezekiel", 27: "Daniel",
    28: "Hosea", 29: "Joel", 30: "Amos", 31: "Obadiah", 32: "Jonah",
    33: "Micah", 34: "Nahum", 35: "Habakkuk", 36: "Zephaniah", 37: "Haggai",
    38: "Zechariah", 39: "Malachi", 40: "Matthew", 41: "Mark", 42: "Luke",
    43: "John", 44: "Acts", 45: "Romans", 46: "1 Corinthians",
    47: "2 Corinthians", 48: "Galatians", 49: "Ephesians", 50: "Philippians",
    51: "Colossians", 52: "1 Thessalonians", 53: "2 Thessalonians",
    54: "1 Timothy", 55: "2 Timothy", 56: "Titus", 57: "Philemon",
    58: "Hebrews", 59: "James", 60: "1 Peter", 61: "2 Peter",
    62: "1 John", 63: "2 John", 64: "3 John", 65: "Jude", 66: "Revelation",
}

PROPHETS_NAMES = {k: v for k, v in ALL_BOOK_NAMES.items() if k <= 39}

FULFILLED_NAMES = {k: v for k, v in ALL_BOOK_NAMES.items() if k >= 40}


def clean_corpus(series):
    return series.astype(str).str.lower().apply(
        lambda x: " ".join(w for w in x.split() if w not in STOP_WORDS)
    )


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("t_bbe.csv").dropna()
    df["Book Name"] = df["b"].map(ALL_BOOK_NAMES)
    df["corpus"] = clean_corpus(df["t"])
    return df, ALL_BOOK_NAMES


@st.cache_data
def load_prophets():
    df = pd.read_csv("t_bbe.csv").dropna()
    df["Book Name"] = df["b"].map(PROPHETS_NAMES)
    df["corpus"] = clean_corpus(df["t"])
    return df, PROPHETS_NAMES


@st.cache_data
def load_fulfilled():
    df = pd.read_csv("t_bbe.csv").dropna()
    df["Book Name"] = df["b"].map(FULFILLED_NAMES)
    df["corpus"] = clean_corpus(df["t"])
    return df, FULFILLED_NAMES


data, book_names = load_data()
prophets, prophets_names = load_prophets()
fulfilled, fulfilled_names = load_fulfilled()

book_numbers = {v: k for k, v in book_names.items()}
prophets_book_numbers = {v: k for k, v in prophets_names.items()}

# ── TF-IDF matrices (sparse — never materialise full N×N similarity) ─────────
# Keeping the matrices sparse avoids the 7.5 GB dense cosine_similarity call.
# Similarities are computed on-demand for one row at a time instead.

@st.cache_resource
def build_tfidf_matrix(corpus_series):
    """Fit and return a sparse TF-IDF matrix + the fitted vectorizer."""
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(corpus_series)
    return matrix, vec


tfidf_matrix, tfidf_vec = build_tfidf_matrix(data["corpus"])
tfidf_matrix_prophecy, _ = build_tfidf_matrix(fulfilled["corpus"])


def get_top_similar(matrix, idx: int, top_n: int):
    """
    Compute cosine similarity between row `idx` and all other rows,
    using sparse dot product — memory stays flat regardless of corpus size.
    """
    row = matrix[idx]                          # (1, n_features) sparse
    scores = (matrix @ row.T).toarray().flatten()  # (n_verses,)
    scores[idx] = 0                            # exclude the input verse itself
    top_idx = np.argsort(scores)[::-1][:top_n]
    return top_idx, scores[top_idx]


# ── TF-IDF co-occurrence query expansion ─────────────────────────────────────
# Reuses the sparse TF-IDF matrix to find words that co-occur with query terms.
# Operates on word columns (sparse), never builds a dense word×word matrix.

@st.cache_resource
def build_cooccurrence_index():
    """
    Returns a small vocab TF-IDF matrix and feature names for co-occurrence lookup.
    max_features=3000 keeps the word×word dot products cheap.
    """
    vec = TfidfVectorizer(max_features=3000, min_df=3)
    matrix = vec.fit_transform(data["corpus"])  # sparse (n_verses, 3000)
    return matrix, vec.get_feature_names_out()


def get_similar_terms(query: str, top_n: int = 5) -> list:
    matrix, feature_names = build_cooccurrence_index()
    name_to_idx = {w: i for i, w in enumerate(feature_names)}

    query_words = [
        w.lower() for w in query.split()
        if w.lower() not in STOP_WORDS and w.isalpha()
    ]
    known = [w for w in query_words if w in name_to_idx]
    if not known:
        return []

    # For each known word, get its column (sparse), compute dot with all other
    # columns — result is a dense (3000,) vector of co-occurrence scores
    scores = np.zeros(len(feature_names))
    for w in known:
        col = matrix.getcol(name_to_idx[w])          # sparse (n_verses, 1)
        scores += (matrix.T @ col).toarray().flatten()  # (3000,)
        scores[name_to_idx[w]] = 0  # exclude self

    top_idx = np.argsort(scores)[::-1][:top_n]
    return [feature_names[i] for i in top_idx if scores[i] > 0]
# ── Semantic embeddings via HuggingFace Inference API (no local model) ────────
# Uses sentence-transformers/all-MiniLM-L6-v2 hosted on HF — same model, zero
# local weight. Requires HF_TOKEN in Streamlit secrets.

@st.cache_data(show_spinner=False)
def load_verse_embeddings() -> np.ndarray:
    """
    Downloads precomputed MiniLM embeddings from Google Drive on first run,
    then caches them in Streamlit's data cache for the session.
    Shape: (31103, 384), float32, L2-normalised.
    """
    import io, urllib.request
    emb_file = "bible_embeddings.npy"
    if not os.path.exists(emb_file):
        with st.spinner("Downloading verse embeddings (first run only)…"):
            # gdown is gone — use a direct urllib download instead
            gdrive_url = (
                "https://drive.google.com/uc?export=download"
                "&id=1-z5RDrWKn13t65PmsWb4FhOGyRcJbOpB"
            )
            urllib.request.urlretrieve(gdrive_url, emb_file)
    emb = np.load(emb_file, allow_pickle=True).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    return emb / norms  # L2-normalised: dot product == cosine similarity


def get_query_embedding(text: str) -> np.ndarray:
    """
    Encodes a query string via the HuggingFace Inference API (feature-extraction).
    Returns a normalised float32 vector of shape (1, 384).
    """
    hf_token = st.secrets.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

    try:
        resp = requests.post(
            api_url,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        vec = np.array(resp.json(), dtype=np.float32)
        # HF returns shape (1, 384) or (384,) — normalise either way
        vec = vec.reshape(1, -1)
        vec = vec / np.linalg.norm(vec, keepdims=True).clip(min=1e-9)
        return vec
    except requests.exceptions.Timeout:
        st.error("⚠️ Embedding model is loading on HuggingFace. Try again in ~20 seconds.")
        return None
    except Exception as e:
        st.error(f"⚠️ Embedding error: {e}")
        return None


def find_similar_verses(query: str, top_n: int = 5) -> pd.DataFrame:
    embeddings = load_verse_embeddings()
    q_vec = get_query_embedding(query)
    if q_vec is None:
        return pd.DataFrame(columns=["Book Name", "c", "v", "t", "Similarity"])
    scores = (embeddings @ q_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_n]
    results = data.iloc[top_idx][["Book Name", "c", "v", "t"]].copy()
    results["Similarity"] = scores[top_idx]
    results.columns = ["Book Name", "c", "v", "t", "Similarity"]
    return results.reset_index(drop=True)



# ── Tab 2: TF-IDF verse recommender ──────────────────────────────────────────
def top_verse(input_book, input_chapter, input_verse, top_n=10):
    try:
        book_num = str(book_numbers.get(input_book, ""))
        locator = data.loc[
            (data["b"].astype(str) == book_num)
            & (data["c"].astype(str) == str(input_chapter))
            & (data["v"].astype(str) == str(input_verse))
        ]
        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])
        idx = locator.index[0]
        sim_idx, sim_val = get_top_similar(tfidf_matrix, idx, top_n)
        rec = data.iloc[sim_idx].copy()
        rec["Similarity Score"] = sim_val
        rec = rec[["Book Name", "c", "v", "t", "Similarity Score"]]
        rec.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        return rec
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])


# ── Tab 5: Prophecy recommender ───────────────────────────────────────────────
def top_verse_prophecy(input_book, input_chapter, input_verse, top_n=10):
    try:
        book_num = str(prophets_book_numbers.get(input_book, ""))
        locator = prophets.loc[
            (prophets["b"].astype(str) == book_num)
            & (prophets["c"].astype(str) == str(input_chapter))
            & (prophets["v"].astype(str) == str(input_verse))
        ]
        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])
        idx = locator.index[0]
        # Use the fulfilled matrix index — prophets idx maps into fulfilled rows
        # via the shared t_bbe.csv row numbers
        sim_idx, sim_val = get_top_similar(tfidf_matrix_prophecy, idx, top_n)
        rec = fulfilled.iloc[sim_idx].copy()
        rec["Similarity Score"] = sim_val
        rec = rec[["Book Name", "c", "v", "t", "Similarity Score"]]
        rec.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        return rec[rec["Book"].notna()]
    except Exception as e:
        st.error(f"Error in prophecy recommendation: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])


# ── Tab 3: RAG summary via HuggingFace Inference API (replaces distilgpt2) ───
def rag_generate(query: str, results_df: pd.DataFrame) -> str:
    """
    Generates a thematic reflection using the HuggingFace Inference API.
    Requires HF_TOKEN set in Streamlit secrets (st.secrets["HF_TOKEN"]).
    Falls back gracefully if the token is missing.
    """
    hf_token = st.secrets.get("HF_TOKEN", "")
    if not hf_token:
        return (
            "⚠️ No Hugging Face token found. Add `HF_TOKEN` to your Streamlit secrets "
            "to enable AI-generated reflections."
        )

    verses = "\n".join(results_df["t"].tolist())
    prompt = (
        f"Based on the following Bible verses:\n\n{verses}\n\n"
        f"Reflect on this theme: '{query}'\n\nReflection:"
    )

    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "do_sample": True, "temperature": 0.7},
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result:
            generated = result[0].get("generated_text", "")
            # Return only the new text after the prompt
            return generated[len(prompt):].strip() or generated.strip()
        return str(result)
    except requests.exceptions.Timeout:
        return "⚠️ The model is loading on HuggingFace servers. Try again in ~20 seconds."
    except Exception as e:
        return f"⚠️ Error generating reflection: {e}"


# ── Tab 4: Image generation via Pollinations.ai (free, no token needed) ───────
def generate_image(prompt: str, width: int, height: int):
    """
    Generates an image via Pollinations.ai — completely free, no API key required.
    Returns a PIL Image or None on failure.
    """
    import io
    from PIL import Image

    encoded_prompt = requests.utils.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width={width}&height={height}&nologo=true&seed={hash(prompt) % 9999}"
    )

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    except requests.exceptions.Timeout:
        st.warning("⚠️ Image generation timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"⚠️ Image generation error: {e}")
        return None


# ── UI ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Table of Contents", "Verse Recommender", "Semantic Search Recommender", "Create Image", "Prophecy"]
)

# ── Tab 1: Table of contents ──────────────────────────────────────────────────
with tab1:
    st.title("📖 Bible Application")
    st.write("**Tab 2: Verse Recommender** ➡️ Find other verses similar to a verse selection from the Old and New Testament")
    st.write("**Tab 3: Semantic Search Recommender** ➡️ Find Bible Verses most similar to user-input words or phrases")
    st.write("**Tab 4: Create Image** ➡️ Select a verse or passage to create an image")
    st.write("**Tab 5: Prophecy** ➡️ Select an Old Testament Prophecy to see where it was fulfilled in the New Testament")

# ── Tab 2: TF-IDF verse-to-verse recommender ─────────────────────────────────
with tab2:
    st.title("📖 Bible Verse Recommender")
    st.write("Find verses similar to your selection from the Old and New Testament.")

    with st.form("verse_input"):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_book = st.selectbox("Select Book", list(book_names.values()))
        with col2:
            input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
        with col3:
            input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1, step=1)

        top_n = st.slider("Number of Similar Verses", min_value=1, max_value=50, value=10, step=5)
        submitted = st.form_submit_button("Find Similar Verses")

    if submitted:
        searched_verse = data.loc[
            (data["Book Name"] == input_book)
            & (data["c"].astype(str) == str(input_chapter))
            & (data["v"].astype(str) == str(input_verse))
        ]
        if not searched_verse.empty:
            st.write(f"**Input Verse:** {searched_verse.iloc[0]['t']}")
            st.write("### 🔍 Similar Verses:")
            st.table(top_verse(input_book, input_chapter, input_verse, top_n))
        else:
            st.warning("Verse not found. Check the chapter and verse numbers.")

# ── Tab 3: Semantic search + Word2Vec expansion + RAG summary ─────────────────
with tab3:
    st.title("📖 Bible Verse Similarity Finder")

    query = st.text_input("Enter a phrase or verse:", "Love your neighbor as yourself")
    top_n_t3 = st.slider("Number of similar verses:", min_value=1, max_value=50, value=10, step=5)

    # TF-IDF co-occurrence query expansion
    similar_terms = get_similar_terms(query, top_n=5)

    if similar_terms:
        st.write("🔍 Similar terms to expand your search:")
        selected_terms = st.multiselect("Add terms to search", options=similar_terms)
        expanded_query = query + (" " + " ".join(selected_terms) if selected_terms else "")
    else:
        expanded_query = query

    if st.button("Find Similar Verses"):
        with st.spinner("Searching…"):
            results = find_similar_verses(expanded_query, top_n_t3)

        st.write("### 🔍 Similar Verses:")
        for _, row in results.iterrows():
            st.write(f"**Book:** {row['Book Name']} | **Chapter:** {row['c']} | **Verse:** {row['v']}")
            st.write(f"**Text:** {row['t']} *(Similarity: {row['Similarity']:.2f})*")

        if not results.empty:
            with st.spinner("Generating reflection…"):
                summary = rag_generate(expanded_query, results)
            st.markdown("### 🧠 RAG Summary")
            st.write(summary)

# ── Tab 4: Image generation ───────────────────────────────────────────────────
with tab4:
    st.title("🖼️ Bible Passage Text-to-Image Generator")
    st.write("Select a verse or passage to create an image.")

    with st.form("image_input"):
        col1, col2 = st.columns(2)
        with col1:
            in_book = st.selectbox("Select Book", list(book_names.values()))
            in_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
        col3a, col3b = st.columns(2)
        with col3a:
            start_verse = st.number_input("Start Verse", min_value=1, max_value=176, value=1, step=1)
        with col3b:
            end_verse = st.number_input("End Verse", min_value=1, max_value=176, value=1, step=1)

        col4a, col4b = st.columns(2)
        with col4a:
            style = st.selectbox(
                "Select Style",
                ["realistic", "oil painting", "digital art", "sketch", "fantasy art"],
            )
        with col4b:
            resolution = st.selectbox("Image Resolution", ["512x512", "768x768"])

        img_submitted = st.form_submit_button("Generate Image")

    if img_submitted:
        selected_verses = data.loc[
            (data["Book Name"] == in_book)
            & (data["c"].astype(str) == str(in_chapter))
            & (data["v"].astype(int) >= start_verse)
            & (data["v"].astype(int) <= end_verse)
        ]
        if not selected_verses.empty:
            passage = " ".join(selected_verses["t"].tolist())
            truncated = passage[:300]
            prompt = f"{truncated}, style: {style}"
            width, height = map(int, resolution.split("x"))

            st.write(f"**Input Passage:** {passage}")
            st.write("### 🖼️ Generated Image:")
            with st.spinner("Generating image…"):
                image = generate_image(prompt, width, height)
            if image:
                st.image(image, width=600)
        else:
            st.warning("Passage not found.")

# ── Tab 5: Prophecy fulfillment ───────────────────────────────────────────────
with tab5:
    st.title("📖 Prophecy Verse Search")
    st.write("Select an Old Testament Prophecy to see where it was fulfilled in the New Testament.")
    st.info("Enter a Book, Chapter and Verse ➡️ click 'Find Prophecy Fulfillment Verses' to find New Testament Bible verses where Old Testament Prophecies were fulfilled.")
    st.info("This application works best when a verse containing a prophecy is selected.")
    st.info("Check out the links immediately below to find prophetic Old Testament verses:")
    st.markdown("[Review Prophecies and Corresponding Fulfillment Verses](https://www.jesusfilm.org/blog/old-testament-prophecies/)")
    st.markdown("[Example List of Prophecies](https://www.newtestamentchristians.com/bible-study-resources/351-old-testament-prophecies-fulfilled-in-jesus-christ/)")

    with st.form("prophecy_input"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_input_book = st.selectbox("Select Book", list(prophets_names.values()))
        with col2:
            p_input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
        with col3:
            p_input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1, step=1)

        p_submitted = st.form_submit_button("➡️ Find Prophecy Fulfillment Verses")

    if p_submitted:
        results = top_verse_prophecy(p_input_book, p_input_chapter, p_input_verse, top_n=10)
        searched_verse = prophets.loc[
            (prophets["Book Name"] == p_input_book)
            & (prophets["c"].astype(str) == str(p_input_chapter))
            & (prophets["v"].astype(str) == str(p_input_verse))
        ]
        if not searched_verse.empty:
            st.write(f"**Input Verse:** {searched_verse.iloc[0]['t']}")
            st.write("### 🔍 Corresponding Verses:")
            for _, row in results.iterrows():
                st.write(f"**Book:** {row['Book']} | **Chapter:** {row['Chapter']} | **Verse:** {row['Verse']}")
                st.write(f"**Text:** {row['Text']} *(Similarity: {row['Similarity Score']:.2f})*")
        else:
            st.warning("Verse not found.")