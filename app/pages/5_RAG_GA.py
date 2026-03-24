import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG and QA", layout="wide")

st.title("🤖 RAG and Question Answering")
st.markdown(
    "Ask a question about the insurance reviews. "
    "The application retrieves the most relevant reviews and generates a synthetic answer."
)

# --- Chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "reviews_clean.csv")


# --- Chargement données ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["avis_clean"]).copy()
    df["avis_clean"] = df["avis_clean"].astype(str)
    return df.reset_index(drop=True)


# --- TF-IDF sur tout le dataset ---
@st.cache_resource
def build_tfidf(_df):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(_df["avis_clean"])
    return vectorizer, matrix


# --- Fonction de recherche ---
def retrieve_reviews(question, df, vectorizer, tfidf_matrix, k=5):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    result = df.copy()
    result["similarity_score"] = scores
    return result.sort_values("similarity_score", ascending=False).head(k)


# --- Génération réponse RAG simple ---
def generate_rag_answer(question, retrieved_df):
    if len(retrieved_df) == 0:
        return "No relevant reviews were found."

    avg_rating = None
    if "note" in retrieved_df.columns and retrieved_df["note"].notna().sum() > 0:
        avg_rating = round(retrieved_df["note"].mean(), 2)

    insurers = (
        retrieved_df["assureur"].dropna().unique().tolist()
        if "assureur" in retrieved_df.columns
        else []
    )

    parts = [f"**Question:** {question}", ""]

    if avg_rating is not None:
        parts.append(f"📊 Average rating among retrieved reviews: **{avg_rating}/5**")

    if insurers:
        parts.append(
            "🏢 Most relevant insurer(s): **" + ", ".join(map(str, insurers[:5])) + "**"
        )

    parts.append("")
    parts.append("📝 Key points from the most relevant reviews:")

    for i, (_, row) in enumerate(retrieved_df.head(3).iterrows(), start=1):
        avis = str(row.get("avis", row.get("avis_clean", "")))[:300]
        insurer = row.get("assureur", "?")
        note = row.get("note", "?")
        parts.append(f"\n**Review {i}** ({insurer} — {note}★): {avis}...")

    return "\n\n".join(parts)


# --- Extraction QA simple ---
def extract_qa_answer(question, retrieved_df):
    question_words = set(question.lower().split())

    best_sentence = None
    best_score = 0
    best_row = None

    for _, row in retrieved_df.iterrows():
        avis = str(row.get("avis", row.get("avis_clean", "")))
        sentences = [s.strip() for s in avis.split(".") if len(s.strip()) > 20]

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
                best_row = row

    return best_sentence, best_row


# =====================
# CHARGEMENT INITIAL
# =====================
df_full = load_data()
vectorizer, tfidf_matrix = build_tfidf(df_full)

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Mode", ["RAG", "QA"])
top_k = st.sidebar.slider("Number of retrieved reviews", 3, 10, 5)

insurer_options = ["All"] + sorted(df_full["assureur"].dropna().unique().tolist())
selected_insurer = st.sidebar.selectbox("Filter by insurer", insurer_options)

rating_options = ["All", "1", "2", "3", "4", "5"]
selected_rating = st.sidebar.selectbox("Filter by rating", rating_options)

# --- Filtrage ---
df_filtered = df_full.copy()
if selected_insurer != "All":
    df_filtered = df_filtered[df_filtered["assureur"] == selected_insurer]
if selected_rating != "All":
    df_filtered = df_filtered[df_filtered["note"] == int(selected_rating)]

st.sidebar.markdown(f"📄 **{len(df_filtered)} reviews** available")

# --- Interface principale ---
st.subheader("💬 Ask a question")

# Questions exemples cliquables
st.markdown("**Example questions:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏆 Best insurer for customer service?"):
        st.session_state["question"] = "best insurer customer service"
with col2:
    if st.button("😤 Main complaints about claims?"):
        st.session_state["question"] = "complaints claims processing slow"
with col3:
    if st.button("💰 Which insurer has best pricing?"):
        st.session_state["question"] = "best pricing cheap insurer"

question = st.text_input(
    "Your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. Which insurer has the best customer service?"
)

if st.button("🔍 Run", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif len(df_filtered) == 0:
        st.warning("No reviews available for the selected filters.")
    else:
        # Rebuild TF-IDF sur le df filtré
        with st.spinner("Searching relevant reviews..."):
            vec_filtered = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            mat_filtered = vec_filtered.fit_transform(df_filtered["avis_clean"])
            retrieved_df = retrieve_reviews(
                question, df_filtered, vec_filtered, mat_filtered, k=top_k
            )

        # --- Affichage des reviews récupérés ---
        st.subheader(f"📚 Top {top_k} Retrieved Reviews")
        for i, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
            insurer = row.get("assureur", "?")
            note = row.get("note", "?")
            score = row["similarity_score"]
            label = f"Review {i} | {insurer} | {note}★ | relevance: {score:.3f}"
            with st.expander(label):
                st.markdown("**Original review:**")
                st.write(row.get("avis", ""))
                if row.get("produit"):
                    st.caption(f"Product: {row['produit']}")

        st.divider()

        # --- Mode RAG ---
        if mode == "RAG":
            st.subheader("🤖 Generated Answer (RAG)")
            answer = generate_rag_answer(question, retrieved_df)
            st.markdown(answer)

        # --- Mode QA ---
        else:
            st.subheader("🎯 Extractive QA Answer")
            best_sentence, best_row = extract_qa_answer(question, retrieved_df)

            if best_sentence and best_row is not None:
                st.success(f"**Answer:** {best_sentence}")
                st.write(f"**Insurer:** {best_row.get('assureur', '?')}")
                st.write(f"**Rating:** {best_row.get('note', '?')}★")
                st.markdown("**Source review:**")
                st.write(best_row.get("avis", ""))
            else:
                st.warning("No specific answer could be extracted.")