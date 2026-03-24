import os
import streamlit as st
import joblib
import pandas as pd
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Explanation", layout="wide")

st.title("Prediction Explanation")

st.markdown("Enter a customer review to understand the predicted rating and its confidence.")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_lr.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_explainer():
    return LimeTextExplainer(class_names=["1★", "2★", "3★", "4★", "5★"])

model = load_model()
explainer = load_explainer()

review_input = st.text_area("Review text", height=180)

if st.button("Explain Prediction"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        pred = model.predict([review_input])[0]

        st.subheader("Predicted Rating")
        st.markdown(f"### {pred} / 5")

        if pred >= 4:
            sentiment = "Positive"
        elif pred == 3:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"

        st.subheader("Predicted Sentiment")
        st.write(sentiment)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([review_input])[0]

            st.subheader("Class Probabilities")

            probs_df = pd.DataFrame({
                "Rating": [f"{i+1}★" for i in range(len(proba))],
                "Probability": [float(p) for p in proba]
            })

            st.bar_chart(probs_df.set_index("Rating"))

            max_proba = float(max(proba))

            st.subheader("Model Confidence")

            if max_proba >= 0.70:
                confidence_text = "High confidence"
            elif max_proba >= 0.50:
                confidence_text = "Moderate confidence"
            else:
                confidence_text = "Low confidence"

            st.write(confidence_text)

            st.subheader("Interpretation")

            if pred >= 4:
                explanation = (
                    "The review is predicted as positive because the model assigns the highest probability "
                    "to high ratings. This suggests that the wording of the review is more similar to reviews "
                    "expressing satisfaction, trust, or positive customer experience."
                )
            elif pred == 3:
                explanation = (
                    "The review is predicted as neutral because the model does not strongly favor either low "
                    "or high ratings. This usually indicates a mixed or moderate opinion."
                )
            else:
                explanation = (
                    "The review is predicted as negative because the model assigns the highest probability "
                    "to low ratings. This suggests that the wording of the review is closer to complaints, "
                    "service dissatisfaction, delays, or negative customer experiences."
                )

            st.write(explanation)

            st.subheader("LIME Explanation")

            exp = explainer.explain_instance(
                review_input,
                model.predict_proba,
                num_features=10
            )

            st.markdown("Top words influencing the prediction:")
            lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Weight"])
            st.dataframe(lime_df)

            st.markdown("Interactive explanation:")
            components.html(exp.as_html(), height=800, scrolling=True)