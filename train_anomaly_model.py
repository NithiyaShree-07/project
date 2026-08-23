import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

def train_anomaly():
    # 1. Load data statistics from scaler
    if not os.path.exists("ml/scaler.pkl"):
        print("Error: scaler.pkl not found. Please train risk model first.")
        return
        
    scaler = joblib.load("ml/scaler.pkl")
    mean_dosage = scaler["mean_dosage"]
    std_dosage = scaler["std_dosage"]
    mean_frequency = scaler["mean_frequency"]
    std_frequency = scaler["std_frequency"]
    mean_medications = scaler["mean_medications"]
    std_medications = 2.5 # Approximate historical std deviation for medications count
    
    # 2. Generate a synthetic dataset with normal patterns and some injected anomalies
    print("Generating Anomaly Dataset...")
    np.random.seed(42)
    n_normal = 250
    n_anomalous = 30
    
    # Normal data
    normal_meds = np.random.randint(1, 7, size=n_normal)
    normal_dosage = np.random.normal(400, 150, size=n_normal)
    normal_dosage = np.clip(normal_dosage, 50, 1000).astype(int)
    normal_freq = np.random.randint(1, 7, size=n_normal)
    normal_delay = np.random.randint(0, 5, size=n_normal)
    normal_interactions = np.random.choice([0, 1], p=[0.9, 0.1], size=n_normal)
    
    # Anomalous data (outliers)
    anom_meds = np.random.randint(10, 18, size=n_anomalous) # polypharmacy
    anom_dosage = np.random.randint(3000, 6000, size=n_anomalous) # massive dosage
    anom_freq = np.random.randint(15, 24, size=n_anomalous) # extremely frequent refills
    anom_delay = np.random.randint(20, 45, size=n_anomalous) # massive delay
    anom_interactions = np.random.randint(4, 8, size=n_anomalous) # high drug-drug interactions
    
    # Combine
    num_medications = np.concatenate([normal_meds, anom_meds])
    dosage_amount = np.concatenate([normal_dosage, anom_dosage])
    prescription_frequency = np.concatenate([normal_freq, anom_freq])
    refill_delay_days = np.concatenate([normal_delay, anom_delay])
    interaction_count = np.concatenate([normal_interactions, anom_interactions])
    
    # Calculate mathematical features
    dosage_zscore = (dosage_amount - mean_dosage) / std_dosage
    prescription_frequency_zscore = (prescription_frequency - mean_frequency) / std_frequency
    medicine_count_zscore = (num_medications - mean_medications) / std_medications
    
    df = pd.DataFrame({
        "num_medications": num_medications,
        "dosage_amount": dosage_amount,
        "prescription_frequency": prescription_frequency,
        "refill_delay_days": refill_delay_days,
        "interaction_count": interaction_count,
        "dosage_zscore": dosage_zscore,
        "prescription_frequency_zscore": prescription_frequency_zscore,
        "medicine_count_zscore": medicine_count_zscore,
        "label": ["Normal"] * n_normal + ["Anomaly"] * n_anomalous
    })
    
    # Save the processed dataset
    df.to_csv("data/processed_dataset.csv", index=False)
    print("Anomaly dataset loaded")
    print("Features created")
    
    # 3. Fit Isolation Forest
    feature_cols = [
        "num_medications",
        "dosage_amount",
        "prescription_frequency",
        "refill_delay_days",
        "interaction_count",
        "dosage_zscore",
        "prescription_frequency_zscore",
        "medicine_count_zscore"
    ]
    X = df[feature_cols]
    
    print("Training IsolationForest Anomaly Detector...")
    # contamination specifies the expected proportion of outliers in the training set
    model = IsolationForest(contamination=0.10, random_state=42)
    model.fit(X)
    print("Model trained")
    
    # Predict (-1 for anomaly, 1 for normal)
    preds = model.predict(X)
    df["pred_score"] = model.decision_function(X)
    df["prediction"] = np.where(preds == -1, "Anomaly", "Normal")
    
    # Evaluate
    # Actual Anomalies vs Predicted Anomalies
    correct_normal = len(df[(df["label"] == "Normal") & (df["prediction"] == "Normal")])
    correct_anom = len(df[(df["label"] == "Anomaly") & (df["prediction"] == "Anomaly")])
    
    acc = (correct_normal + correct_anom) / len(df)
    print(f"Anomaly Detection Accuracy: {acc * 100:.2f}%")
    print(f"Correctly flagged anomalies: {correct_anom}/{n_anomalous}")
    
    # Save model
    joblib.dump(model, "ml/anomaly_model.pkl")
    print("Model saved successfully")
    
    # Save anomaly metrics
    metrics = {
        "accuracy": round(acc * 100, 1),
        "total_anomalies_flagged": int((preds == -1).sum()),
        "contamination_rate": 0.10
    }
    with open("ml/anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_anomaly()
