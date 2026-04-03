from pathlib import Path
import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MAX_TEXT_LENGTH = 10000

vectorizer = None
model = None
label_encoder = None

def load_artifacts():
    global vectorizer, model, label_encoder

    vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

def prepare_text(text):
    text = str(text)
    return text[:MAX_TEXT_LENGTH]

def predict(text):

    text = prepare_text(text)
    text_vec = vectorizer.transform([text])
    probs = model.predict_proba(text_vec)[0]

    top_k_indicies = np.argsort(probs)[::-1]
    top_k_probs = probs[top_k_indicies]
    top_k_labels = label_encoder.inverse_transform(top_k_indicies)

    best_idx = top_k_indicies[0]

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