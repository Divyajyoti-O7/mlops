import os

with open("mlruns/drift_report.txt", "r") as f:
    report = f.read()

if "RETRAIN REQUIRED" in report:
    print("Retraining triggered due to drift...")
    os.system("python src/train.py")
else:
    print("No drift detected. Retraining skipped.")
