import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="RAG and QA", layout="wide")

st.title("RAG and Question Answering")

st.markdown(
    "Ask a question about the insurance reviews. "
    "The application retrieves the most relevant reviews and either generates a synthetic answer "
    "or extracts an answer directly from the retrieved texts."
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "reviews_clean.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["avis_clean"]).copy()
    df["avis_clean"] = df["avis_clean"].astype(str)
    return df

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

df = load_data()
sbert_model = load_sbert_model()
qa_pipeline = load_qa_pipeline()

st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Mode",
    ["RAG", "QA"]
)

top_k = st.sidebar.slider("Number of retrieved reviews", min_value=3, max_value=10, value=5)

insurer_options = ["All"] + sorted(df["assureur"].dropna().unique().tolist())
selected_insurer = st.sidebar.selectbox("Insurer", insurer_options)

if selected_insurer != "All":
    df = df[df["assureur"] == selected_insurer].copy()

question = st.text_input("Your question")

def retrieve_reviews(question_text, dataframe, k=5):
    review_texts = dataframe["avis_clean"].tolist()
    review_embeddings = sbert_model.encode(review_texts, show_progress_bar=False)
    question_embedding = sbert_model.encode([question_text], show_progress_bar=False)

    scores = cosine_similarity(question_embedding, review_embeddings)[0]
    retrieved_df = dataframe.copy()
    retrieved_df["similarity_score"] = scores
    retrieved_df = retrieved_df.sort_values("similarity_score", ascending=False).head(k)

    return retrieved_df

def generate_simple_rag_answer(question_text, retrieved_df):
    if len(retrieved_df) == 0:
        return "No relevant reviews were found."

    avg_rating = None
    if "note" in retrieved_df.columns and retrieved_df["note"].notna().sum() > 0:
        avg_rating = round(retrieved_df["note"].mean(), 2)

    insurers = retrieved_df["assureur"].dropna().unique().tolist() if "assureur" in retrieved_df.columns else []
    products = retrieved_df["produit"].dropna().unique().tolist() if "produit" in retrieved_df.columns else []

    sample_points = []
    for _, row in retrieved_df.head(3).iterrows():
        if "avis" in row and pd.notna(row["avis"]):
            sample_points.append(str(row["avis"])[:250])

    answer_parts = []

    answer_parts.append(f"Question: {question_text}")

    if avg_rating is not None:
        answer_parts.append(f"The average rating among the retrieved reviews is {avg_rating}/5.")

    if insurers:
        answer_parts.append("The most relevant reviews mainly concern the following insurer(s): " + ", ".join(map(str, insurers[:5])) + ".")

    if products:
        answer_parts.append("The reviews are mostly related to: " + ", ".join(map(str, products[:5])) + ".")

    if sample_points:
        answer_parts.append("Here are the main ideas found in the most relevant reviews:")
        for point in sample_points:
            answer_parts.append(f"- {point}")

    return "\n\n".join(answer_parts)

if st.button("Run"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    elif len(df) == 0:
        st.warning("No reviews are available for the selected filters.")
    else:
        with st.spinner("Retrieving relevant reviews..."):
            retrieved_df = retrieve_reviews(question, df, k=top_k)

        st.subheader("Retrieved Reviews")

        for i, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
            title_parts = [f"Review {i}"]
            if "assureur" in row and pd.notna(row["assureur"]):
                title_parts.append(f"Insurer: {row['assureur']}")
            if "note" in row and pd.notna(row["note"]):
                title_parts.append(f"Rating: {row['note']}")
            title_parts.append(f"Score: {row['similarity_score']:.3f}")

            with st.expander(" | ".join(title_parts)):
                if "avis" in row and pd.notna(row["avis"]):
                    st.markdown("**Original review:**")
                    st.write(row["avis"])

                if "avis_clean" in row and pd.notna(row["avis_clean"]):
                    st.markdown("**Cleaned review:**")
                    st.write(row["avis_clean"])

                if "produit" in row and pd.notna(row["produit"]):
                    st.write(f"Product: {row['produit']}")

        if mode == "RAG":
            st.subheader("Generated Answer")

            answer = generate_simple_rag_answer(question, retrieved_df)
            st.write(answer)

        else:
            st.subheader("Extractive QA Answer")

            best_answer = None
            best_score = -1

            for _, row in retrieved_df.iterrows():
                context = str(row["avis"])
                try:
                    result = qa_pipeline(question=question, context=context)
                    if result["score"] > best_score and result["answer"].strip() != "":
                        best_score = result["score"]
                        best_answer = {
                            "answer": result["answer"],
                            "score": result["score"],
                            "context": context,
                            "assureur": row["assureur"] if "assureur" in row else None,
                            "note": row["note"] if "note" in row else None
                        }
                except Exception:
                    continue

            if best_answer is not None:
                st.write(f"**Answer:** {best_answer['answer']}")
                st.write(f"**Confidence score:** {best_answer['score']:.3f}")

                if best_answer["assureur"] is not None:
                    st.write(f"**Insurer:** {best_answer['assureur']}")
                if best_answer["note"] is not None:
                    st.write(f"**Rating:** {best_answer['note']}")

                st.markdown("**Source review:**")
                st.write(best_answer["context"])
            else:
                st.warning("No extractive answer could be found.")