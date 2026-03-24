import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Summary", layout="wide")

st.title("Insurer Summary")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # remonte à app/
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "reviews_clean.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.markdown("Select an insurer to display summary statistics and review distribution.")

insurer_list = sorted(df["assureur"].dropna().unique())
selected_insurer = st.selectbox("Select an insurer", insurer_list)

df_insurer = df[df["assureur"] == selected_insurer].copy()

st.subheader(f"Overview for {selected_insurer}")

col1, col2, col3 = st.columns(3)

col1.metric("Number of reviews", len(df_insurer))

if "note" in df_insurer.columns and df_insurer["note"].notna().sum() > 0:
    col2.metric("Average rating", round(df_insurer["note"].mean(), 2))
    positive_rate = (df_insurer["note"] >= 4).mean() * 100
    col3.metric("Positive reviews (%)", round(positive_rate, 1))
else:
    col2.metric("Average rating", "N/A")
    col3.metric("Positive reviews (%)", "N/A")

if "note" in df_insurer.columns and df_insurer["note"].notna().sum() > 0:
    st.subheader("Rating Distribution")

    fig, ax = plt.subplots()
    df_insurer["note"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Stars")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings")

    st.pyplot(fig)

st.subheader("Sample Reviews")

sample_reviews = df_insurer["avis"].dropna().astype(str).head(5).tolist()

for i, review in enumerate(sample_reviews, start=1):
    with st.expander(f"Review {i}"):
        st.write(review)