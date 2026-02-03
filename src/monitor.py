import pandas as pd
import numpy as np
import mlflow
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist

BASELINE_DATA_PATH = "data/raw/Mall_Customers.csv"
INCOMING_DATA_PATH = "data/incoming/Customers_day2.csv"
CENTROIDS_PATH = "artifacts/centroids.npy"

FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

SAFE_FEATURE_NAMES = {
    "Age": "age",
    "Annual Income (k$)": "annual_income",
    "Spending Score (1-100)": "spending_score"
}

KS_PVALUE_THRESHOLD = 0.05
DISTANCE_DRIFT_MULTIPLIER = 1.3

baseline_df = pd.read_csv(BASELINE_DATA_PATH)
incoming_df = pd.read_csv(INCOMING_DATA_PATH)

baseline_X = baseline_df[FEATURES]
incoming_X = incoming_df[FEATURES]


centroids = np.load(CENTROIDS_PATH)

drift_detected = False
drift_results = {}

mlflow.set_experiment("customer_segment_drift")

with mlflow.start_run(run_name="daily_drift_monitoring"):

    for feature in FEATURES:
        stat, p_value = ks_2samp(
            baseline_X[feature],
            incoming_X[feature]
        )

        safe_name = SAFE_FEATURE_NAMES[feature]
        mlflow.log_metric(f"{safe_name}_pvalue", p_value)

        drift_results[feature] = p_value < KS_PVALUE_THRESHOLD
        if drift_results[feature]:
            drift_detected = True

    baseline_distances = cdist(baseline_X, centroids).min(axis=1)
    incoming_distances = cdist(incoming_X, centroids).min(axis=1)

    baseline_avg_distance = baseline_distances.mean()
    incoming_avg_distance = incoming_distances.mean()

    distance_ratio = incoming_avg_distance / baseline_avg_distance
    mlflow.log_metric("cluster_distance_ratio", distance_ratio)

    if distance_ratio > DISTANCE_DRIFT_MULTIPLIER:
        drift_detected = True

    mlflow.log_metric("drift_detected", int(drift_detected))

with open("mlruns/drift_report.txt", "w") as f:
    f.write("CUSTOMER SEGMENT DRIFT REPORT\n")
    f.write("==============================\n\n")

    for feature, drifted in drift_results.items():
        status = "DRIFT DETECTED" if drifted else "OK"
        f.write(f"{feature}: {status}\n")

    f.write("\n")
    f.write(f"Cluster Distance Drift: {'YES' if distance_ratio > DISTANCE_DRIFT_MULTIPLIER else 'NO'}\n")
    f.write("\n")
    f.write(f"FINAL DECISION: {'RETRAIN REQUIRED' if drift_detected else 'MODEL HEALTHY'}\n")

if drift_detected:
    print("⚠️  DRIFT DETECTED → Retraining Recommended")
else:
    print("✅ No Drift Detected → Model Healthy")
