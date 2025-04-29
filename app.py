import streamlit as st
import numpy as np
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocess user input
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Predict function
def predict_news(news):
    news = preprocess(news)
    input_vec = vectorizer.transform([news])
    prediction = model.predict(input_vec)[0]
    confidence = np.max(model.predict_proba(input_vec))
    return prediction, confidence

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("### Enter a news article below and let AI tell you if it's real or fake.")

news_input = st.text_area("✏️ Enter News Article", height=200)

if st.button("🔍 Detect"):
    if news_input.strip():
        prediction, confidence = predict_news(news_input)
        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"**Prediction:** {result}\n\n**Confidence:** {confidence*100:.2f}%")
    else:
        st.warning("Please enter a news article.")
