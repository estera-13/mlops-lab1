# inference.py
import joblib
from training import MODEL_PATH

# Globalna zmienna modelu – ładowana raz
_model = None


def load_model():
    """Load the trained model once and keep it in memory."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(input_data: dict) -> str:
    """
    Predict the Iris class for given input data.

    input_data: dict with keys:
        sepal_length, sepal_width, petal_length, petal_width

    Returns predicted class name as string.
    """
    model = load_model()
    # Zamień dict na listę cech
    features = [
        [
            input_data["sepal_length"],
            input_data["sepal_width"],
            input_data["petal_length"],
            input_data["petal_width"],
        ]
    ]
    prediction_idx = model.predict(features)[0]
    # Nazwy gatunków Iris
    target_names = ["setosa", "versicolor", "virginica"]
    return target_names[prediction_idx]
