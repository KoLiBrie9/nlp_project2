# NLP Project – Customer Reviews Analysis & Interactive App

## Project Overview

This project focuses on analyzing customer reviews using Natural Language Processing  techniques.
It includes data preprocessing, topic modeling, supervised learning and an interactive Streamlit application.

The goal is to extract insights, predict sentiment, and provide explanations and search capabilities on textual data.

---

## Features

* Sentiment prediction using TF-IDF + Logistic Regression
* Text summarization
* Model explanation with LIME
* Search engine over reviews
* Retrieval-Augmented Generation (RAG) module
* Interactive visualizations with Streamlit

---

## Project Structure

```bash
nlp_project2/
├── app/                    # Streamlit application
│   ├── models/             # Models (tfidf_lr.pkl)
│   ├── pages/              # App pages (Prediction, Summary, etc.)
│   ├── streamlit_app.py    # Main app entry point
│
├── data/
│   ├── raw/                # Raw data (Excel files)
│   ├── processed/          # Cleaned datasets
│
├── notebooks/              # Jupyter notebooks (EDA, modeling)
│
├── report/                 # video
│
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd nlp_project2
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app/streamlit_app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas / NumPy
* NLTK / Gensim
* Transformers / Sentence-Transformers
* LIME

---

## Data

The dataset consists of customer reviews collected from multiple sources and translated into English.

* Raw data: Excel files
* Processed data: cleaned and enriched CSV files

---

## Methodology

1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Topic modeling
4. Feature extraction (TF-IDF, embeddings)
5. Supervised learning (Logistic Regression)
6. Model evaluation and interpretation

---

## Future Improvements

* Improve model performance with deep learning
* Enhance the RAG pipeline
* Deploy the application online
* Add more interactive visualizations

---

## Author

Koralie THERESINE & Estelle SOBESKY

---

## Notes

This project was developed as part of an NLP course and demonstrates an end-to-end data science workflow, from raw data to deployment.
