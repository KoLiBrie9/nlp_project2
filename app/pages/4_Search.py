import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Search", layout="wide")

st.title("Review Search")

st.markdown("Search for reviews using keywords and filters.")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "reviews_clean.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Keep only rows with usable review text
df = df.dropna(subset=["avis_clean"]).copy()
df["avis_clean"] = df["avis_clean"].astype(str)

st.sidebar.header("Filters")

# Insurer filter
insurer_options = ["All"] + sorted(df["assureur"].dropna().unique().tolist())
selected_insurer = st.sidebar.selectbox("Insurer", insurer_options)

# Rating filter
if "note" in df.columns:
    rating_options = ["All"] + sorted([int(x) for x in df["note"].dropna().unique()])
    selected_rating = st.sidebar.selectbox("Rating", rating_options)
else:
    selected_rating = "All"

# Apply filters
filtered_df = df.copy()

if selected_insurer != "All":
    filtered_df = filtered_df[filtered_df["assureur"] == selected_insurer]

if selected_rating != "All":
    filtered_df = filtered_df[filtered_df["note"] == selected_rating]

query = st.text_input("Search query")

top_k = st.slider("Number of results", min_value=3, max_value=20, value=5)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a search query.")
    elif len(filtered_df) == 0:
        st.warning("No reviews match the selected filters.")
    else:
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(filtered_df["avis_clean"])

        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix)[0]

        filtered_df = filtered_df.copy()
        filtered_df["similarity_score"] = scores

        results = filtered_df.sort_values("similarity_score", ascending=False).head(top_k)

        st.subheader("Top Results")

        for i, (_, row) in enumerate(results.iterrows(), start=1):
            title_parts = [f"Result {i}"]
            if "assureur" in row and pd.notna(row["assureur"]):
                title_parts.append(f"Insurer: {row['assureur']}")
            if "note" in row and pd.notna(row["note"]):
                title_parts.append(f"Rating: {row['note']}")

            with st.expander(" | ".join(title_parts)):
                st.write(f"Similarity score: {row['similarity_score']:.3f}")

                if "avis" in row and pd.notna(row["avis"]):
                    st.markdown("**Original review:**")
                    st.write(row["avis"])

                if "avis_clean" in row and pd.notna(row["avis_clean"]):
                    st.markdown("**Cleaned review:**")
                    st.write(row["avis_clean"])

                if "produit" in row and pd.notna(row["produit"]):
                    st.write(f"Product: {row['produit']}")

                if "date_publication" in row and pd.notna(row["date_publication"]):
                    st.write(f"Publication date: {row['date_publication']}")