from pathlib import Path
import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
TOP_K = 5

vectorizer = None
model = None
label_encoder = None

def load_artifacts() -> None:
    global vectorizer, model, label_encoder

    vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

def predict(text: str) -> dict:
    if vectorizer is None or model is None or label_encoder is None:
        raise RuntimeError("Artifacts are not loaded. Call load_artifacts() first.")

    text = str(text)
    text_vec = vectorizer.transform([text])
    probs = model.predict_proba(text_vec)[0]

    top_k_limit = min(TOP_K, probs.shape[0])
    top_k_indices = np.argsort(probs)[::-1][:top_k_limit]
    top_k_probs = probs[top_k_indices]

    # Map sorted probability positions to encoded class ids first,
    # then decode ids back to original category labels.
    top_k_class_ids = model.classes_[top_k_indices]
    top_k_labels = label_encoder.inverse_transform(top_k_class_ids.astype(int))

    result = {
        "best_category": top_k_labels[0],
        "best_confidence": float(top_k_probs[0]),
        "predictions": [
            {
                "category": label,
                "confidence": float(prob),
            }
            for label, prob in zip(top_k_labels, top_k_probs)
        ],
    }

    return result

if __name__ == "__main__":
    load_artifacts()

    text = "Noise cancelling over-ear headphones with case"

    result = predict(text)
    print(result)