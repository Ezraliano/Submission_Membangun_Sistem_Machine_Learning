# modelling.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import re
import string
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Download resource nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Function membersihkan teks
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Gabungkan fitur teks
def combine_features(row):
    return f"{row['job_title_clean']} {row['job_description_clean']} {row.get('skills_clean', '')}"

# Fungsi pelatihan model
def train_recommendation_model():
    # Load preprocessed dataset
    base_dir = os.path.abspath(os.path.dirname(__file__))  # lokasi file .py saat ini
    data_path = os.path.join(base_dir, 'dataset_preprocessing', '1000_ml_jobs_us_preprocessed.csv')
    data = pd.read_csv(data_path)

    # Clean ulang jika diperlukan
    data['job_title_clean'] = data['job_title'].apply(clean_text)
    data['job_description_clean'] = data['job_description_text'].apply(clean_text)
    data['skills_clean'] = data['skills'].fillna('').apply(clean_text) if 'skills' in data.columns else ''
    data['combined_text'] = data.apply(combine_features, axis=1)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['combined_text'])

    # Label encoding untuk job_title
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['job_title'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

    # Set experiment
    mlflow.set_experiment("ML_Job_Recommender_Model")

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # Mulai MLflow Run
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[INFO] Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_recommendation_model()
