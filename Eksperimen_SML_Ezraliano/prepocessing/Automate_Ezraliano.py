# automate_NamaAnda.py
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

nltk.download('stopwords')
nltk.download('punkt')

def load_data(file_path):
    return pd.read_csv(file_path)

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

def preprocess_data(data):
    # Drop missing
    data = data.dropna(subset=['job_title', 'job_description_text'])

    # Fill missing in other cols
    fill_cols = ['company_address_region', 'company_address_locality',
                 'company_website', 'company_description', 'seniority_level']
    data[fill_cols] = data[fill_cols].fillna('unknown')

    data['job_title_clean'] = data['job_title'].apply(clean_text)
    data['job_description_clean'] = data['job_description_text'].apply(clean_text)

    if 'skills' in data.columns:
        data['skills_clean'] = data['skills'].fillna('').apply(clean_text)
    else:
        data['skills_clean'] = ''

    data['combined_text'] = data.apply(
        lambda row: f"{row['job_title_clean']} {row['job_description_clean']} {row['skills_clean']}",
        axis=1
    )

    return data

def save_preprocessed_data(data, output_path):
    data.to_csv(output_path, index=False)

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))  # lokasi file .py saat ini
    input_path = os.path.join(base_dir, '..', 'dataset', '1000_ml_jobs_us.csv')
    output_path = os.path.join(base_dir, '..', 'dataset', '1000_ml_jobs_us_preprocessed.csv')

    data = load_data(input_path)
    preprocessed_data = preprocess_data(data)

    save_preprocessed_data(preprocessed_data, output_path)
    print(f"Preprocessing selesai. Data disimpan di: {output_path}")

if __name__ == '__main__':
    main()
