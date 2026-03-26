# training.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = "iris_model.joblib"


def load_data():
    """Load the Iris dataset and return features and labels."""
    data = load_iris()
    X = data.data
    y = data.target
    return X, y


def train_model():
    """Train a RandomForestClassifier on the Iris dataset."""
    X, y = load_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def save_model(model, path=MODEL_PATH):
    """Save the trained model to a file using joblib."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    model = train_model()
    save_model(model)
