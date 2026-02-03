import os
os.makedirs("artifacts/model", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans

mlflow.set_experiment("customer_segment_drift")

df = pd.read_csv("data/raw/Mall_Customers.csv")
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

with mlflow.start_run(run_name="baseline_v1"):
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(X)

    joblib.dump(model, "artifacts/model/kmeans.pkl")
    np.save("artifacts/centroids.npy", model.cluster_centers_)
    X.describe().to_csv("artifacts/baseline_stats.csv")

    mlflow.log_param("n_clusters", 5)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("artifacts/baseline_stats.csv")
