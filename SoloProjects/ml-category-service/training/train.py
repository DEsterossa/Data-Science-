from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "ecommerceDataset.csv"
TEXT_COL = 'description'
TARGET_COL = 'category'

RANDOM_STATE =  42
TEST_SIZE = 0.2

MAX_TEXT_LENGTH = 10000
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 1)
STOP_WORDS = 'english'

def load_dataset(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path)

    df = df.dropna()
    df[TEXT_COL] = df[TEXT_COL].str[:MAX_TEXT_LENGTH]
    df = df[
        (df[TEXT_COL].str.strip() != '') &
        (df[TARGET_COL].str.strip() != '')
    ].reset_index(drop=True)

    if df.empty:
        return ValueError("Dataset is empty after preprocessing.")
    
    return df

def main():
    df = load_dataset(DATA_PATH)
    X = df[TEXT_COL]
    y = df[TARGET_COL]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words=STOP_WORDS
    )

    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model.fit(X_vec, y_encoded)

    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')

    print("Artifacts saved to:", 'models')

if __name__ == '__main__':
    main()
