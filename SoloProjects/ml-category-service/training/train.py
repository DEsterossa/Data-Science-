from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
MODELS_DIR = ROOT_DIR / "models"

TEXT_COL = "title"
TARGET_COL = "categories"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MAX_FEATURES = 30000
NGRAM_RANGE = (1, 2)


def load_dataset(data_path: Path) -> pd.DataFrame:

    df = pd.read_csv(data_path)

    df = df[[TEXT_COL, TARGET_COL]].copy()
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])

    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

    df = df[
        (df[TEXT_COL] != "") &
        (df[TARGET_COL] != "")
    ].copy()

    df = df.drop_duplicates(subset=[TEXT_COL, TARGET_COL]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty after preprocessing.")

    return df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)

    X = df[TEXT_COL]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_vec, y_train_encoded)

    y_pred_encoded = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    macro_f1 = f1_score(y_test_encoded, y_pred_encoded, average="macro")
    weighted_f1 = f1_score(y_test_encoded, y_pred_encoded, average="weighted")

    print("Training finished")
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    joblib.dump(model, MODELS_DIR / "model.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")

    print(f"Artifacts saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()