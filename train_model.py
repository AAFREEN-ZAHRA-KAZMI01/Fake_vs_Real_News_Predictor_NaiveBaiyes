import pandas as pd
import numpy as np
import nltk
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')

# Load CSVs
fake_df = pd.read_csv('Fake.csv', on_bad_lines='skip', encoding='utf-8')
real_df = pd.read_csv('True.csv', on_bad_lines='skip', encoding='utf-8')

fake_df['label'] = 0
real_df['label'] = 1

data = pd.concat([fake_df, real_df], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

# Preprocessing
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

data['content'] = (data['title'] + " " + data['text']).apply(preprocess)

X = data['content']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vect, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test_vect)))
print(classification_report(y_test, model.predict(X_test_vect)))

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
