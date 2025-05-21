import os
import re
import string
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from mlflow.models.signature import infer_signature

# ===================== Setup =====================
nltk.download('punkt')
nltk.download('stopwords')

# ===================== Custom Pyfunc Model =====================
class TfidfRecommenderModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.data = None

    def load_context(self, context):
        import joblib
        # Load model TF-IDF
        self.vectorizer = joblib.load(context.artifacts["tfidf_model"])
        # Load data dan hasil cosine similarity
        self.data = pd.read_csv(context.artifacts["data_csv"])
        tfidf_matrix = self.vectorizer.transform(self.data['combined_text'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def predict(self, context, model_input):
        input_text = model_input.iloc[0]["text"]
        input_vec = self.vectorizer.transform([input_text])
        sim_scores = cosine_similarity(input_vec, self.cosine_sim).flatten()
        top_indices = sim_scores.argsort()[-5:][::-1]  # Top 5 recommendations
        return self.data.iloc[top_indices]['job_title'].tolist()


# ===================== Preprocessing Functions =====================
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
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        recommended_indices = [i[0] for i in sim_scores]
        recommended_titles = data.iloc[recommended_indices]['job_title'].values

        y_true = [1] * top_n
        y_pred = [1 if is_relevant(actual_title, rec) else 0 for rec in recommended_titles]
        relevant_jobs = data[data['job_title'].apply(lambda x: is_relevant(actual_title, x))].index
        relevant_count = len(relevant_jobs) - 1
        if relevant_count == 0:
            continue

        precision = sum(y_pred) / top_n
        recall = sum(y_pred) / relevant_count if relevant_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        'precision_at_5': np.mean(precision_scores),
        'recall_at_5': np.mean(recall_scores),
        'f1_at_5': np.mean(f1_scores)
    }

# ===================== Main Script =====================
def main():
    print("[INFO] Starting training & logging...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("TFIDF_Recommender")

    base_dir = os.path.abspath(os.path.dirname(__file__))
    script_path = os.path.abspath(__file__)
    data_path = os.path.join(base_dir, 'dataset_preprocessing', '1000_ml_jobs_us_preprocessed.csv')

    data = pd.read_csv(data_path)

    # Preprocessing
    data['job_title_clean'] = data['job_title'].apply(clean_text)
    data['job_description_clean'] = data['job_description_text'].apply(clean_text)
    data['skills_clean'] = data['skills'].fillna('').apply(clean_text) if 'skills' in data.columns else ''
    data['combined_text'] = data.apply(combine_features, axis=1)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_text'])

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Evaluate
    metrics = evaluate_recommendations(data, cosine_sim, top_n=5, sample_size=100)
    for metric, value in metrics.items():
        print(f"[METRIC] {metric}: {value:.4f}")

    # Save artifacts
    tfidf_path = os.path.join(base_dir, "tfidf_model.pkl")
    joblib.dump(tfidf, tfidf_path)

    data_csv_path = os.path.join(base_dir, "data.csv")
    data.to_csv(data_csv_path, index=False)

    # Example input for model signature
    input_example = pd.DataFrame({"text": ["example job description"]})
    transformed = tfidf.transform(input_example["text"])
    signature = infer_signature(input_example, transformed)

    # Log model
    with mlflow.start_run():
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        mlflow.pyfunc.log_model(
            artifact_path="tfidf_model_pyfunc",
            python_model=TfidfRecommenderModel(),
            artifacts={
                "tfidf_model": tfidf_path,
                "data_csv": data_csv_path
            },
            input_example=input_example,
            signature=signature,
            code_path=[script_path]
        )

    print("[INFO] Model training complete and logged to MLflow.")

if __name__ == "__main__":
    main()