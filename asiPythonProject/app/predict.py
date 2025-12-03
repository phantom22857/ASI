import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)


def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction
