from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "model" / "kmeans.pkl"
CENTROIDS_PATH = BASE_DIR / "artifacts" / "centroids.npy"

try:
    model = joblib.load(MODEL_PATH)
    centroids = np.load(CENTROIDS_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

app = FastAPI(title="Customer Segmentation API")

class CustomerFeatures(BaseModel):
    features: list[float]


@app.post("/predict")
def predict_customer(data: CustomerFeatures):
    """
    Input: [Age, Annual Income, Spending Score]
    """
    if len(data.features) != 3:
        raise HTTPException(
            status_code=400,
            detail="features must contain exactly 3 values"
        )

    features = np.array(data.features).reshape(1, -1)

    cluster = int(model.predict(features)[0])
    distance = float(cdist(features, centroids).min())

    return {
        "cluster": cluster,
        "distance_to_nearest_cluster": distance
    }
