import pandas as pd
import numpy as np
import os
import re
import string
import nltk
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_score, recall_score, f1_score

nltk.download('punkt')
nltk.download('stopwords')


# =========================== Text Preprocessing ===========================

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

def combine_features(row):
    return f"{row['job_title_clean']} {row['job_description_clean']} {row.get('skills_clean', '')}"

def is_relevant(actual_title, recommended_title):
    actual_words = set(actual_title.lower().split())
    recommended_words = set(recommended_title.lower().split())
    return len(actual_words.intersection(recommended_words)) > 0

def evaluate_recommendations(data, cosine_sim, top_n=5, sample_size=100):
    precision_scores, recall_scores, f1_scores = [], [], []
    sample_indices = np.random.choice(data.index, size=min(sample_size, len(data)), replace=False)

    for idx in sample_indices:
        actual_title = data.iloc[idx]['job_title']
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommended_indices = [i[0] for i in sim_scores]
        recommended_titles = data.iloc[recommended_indices]['job_title'].values

        y_true = [1] * top_n
        y_pred = [1 if is_relevant(actual_title, rec) else 0 for rec in recommended_titles]
        relevant_jobs = data[data['job_title'].apply(lambda x: is_relevant(actual_title, x))].index
        relevant_count = len(relevant_jobs) - 1
        if relevant_count == 0:
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = sum(y_pred) / relevant_count if relevant_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1': np.mean(f1_scores)
    }

# =========================== Main Function ===========================

def tune_recommendation_model():
    # Load dataset
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'dataset_preprocessing', '1000_ml_jobs_us_preprocessed.csv')
    data = pd.read_csv(data_path)

    # Preprocessing
    data['job_title_clean'] = data['job_title'].apply(clean_text)
    data['job_description_clean'] = data['job_description_text'].apply(clean_text)
    data['skills_clean'] = data['skills'].fillna('').apply(clean_text) if 'skills' in data.columns else ''
    data['combined_text'] = data.apply(combine_features, axis=1)

    # Tuning space
    param_grid = {
        'max_df': [0.8, 1.0],
        'min_df': [1, 5],
        'max_features': [3000, 5000],
        'ngram_range': [(1, 1), (1, 2)]
    }

    mlflow.set_experiment("Recommender_Tuning")

    for params in ParameterGrid(param_grid):
        with mlflow.start_run():
            # Log parameter
            for key, value in params.items():
                mlflow.log_param(key, value)

            # TF-IDF
            tfidf = TfidfVectorizer(
                stop_words='english',
                max_df=params['max_df'],
                min_df=params['min_df'],
                max_features=params['max_features'],
                ngram_range=params['ngram_range']
            )
            tfidf_matrix = tfidf.fit_transform(data['combined_text'])

            # Cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Evaluate recommendations
            scores = evaluate_recommendations(data, cosine_sim, top_n=5, sample_size=100)

            # Log metrics
            mlflow.log_metric("precision_at_5", scores['precision'])
            mlflow.log_metric("recall_at_5", scores['recall'])
            mlflow.log_metric("f1_at_5", scores['f1'])

            print(f"[INFO] Done tuning: {params} | F1@5: {scores['f1']:.4f}")

if __name__ == "__main__":
    tune_recommendation_model()
