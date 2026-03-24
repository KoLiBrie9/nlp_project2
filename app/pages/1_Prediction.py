import os
import streamlit as st
import joblib

st.set_page_config(page_title="Prediction", layout="wide")

st.title("Review Rating Prediction")
st.markdown("Enter a customer review to predict its star rating.")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # remonte à app/
MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_lr.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

review_input = st.text_area("Review text", height=150)

if st.button("Predict"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        pred = model.predict([review_input])[0]

        st.subheader("Predicted Rating")
        st.markdown(f"### {pred} / 5")
        st.write("".join(["⭐"] * int(pred)))

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([review_input])[0]

            st.subheader("Class Probabilities")

            probs_dict = {
                f"{i+1}★": float(proba[i])
                for i in range(len(proba))
            }

            st.bar_chart(probs_dict)

        if pred >= 4:
            sentiment = "Positive"
        elif pred == 3:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"

        st.subheader("Predicted Sentiment")
        st.write(sentiment)