import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
os.makedirs("ml", exist_ok=True)

def generate_risk_dataset(filename="data/prescription_dataset.csv", num_samples=350, seed=42):
    """
    Generates synthetic training dataset for NALAM risk classification.
    """
    np.random.seed(seed)
    
    # 1. Independent patient and prescription features
    patient_age = np.random.randint(18, 90, size=num_samples)
    gender = np.random.choice(["M", "F"], size=num_samples)
    num_medications = np.random.randint(1, 12, size=num_samples)
    
    # Dosage amount in mg
    dosage_amount = np.random.normal(500, 300, size=num_samples)
    dosage_amount = np.clip(dosage_amount, 20, 2000).astype(int)
    
    dosage_frequency = np.random.randint(1, 5, size=num_samples) # times per day
    treatment_duration = np.random.choice([7, 14, 30, 90, 180], size=num_samples)
    allergy_flag = np.random.binomial(1, 0.25, size=num_samples)
    chronic_disease_count = np.random.randint(0, 4, size=num_samples)
    
    # Interaction count correlated with medication count
    interaction_count = []
    for num_med in num_medications:
        if num_med <= 2:
            ic = np.random.choice([0, 1], p=[0.9, 0.1])
        elif num_med <= 5:
            ic = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        else:
            ic = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.1, 0.1])
        interaction_count.append(ic)
    interaction_count = np.array(interaction_count)
    
    medication_adherence = np.random.normal(80, 15, size=num_samples)
    medication_adherence = np.clip(medication_adherence, 30, 100).astype(int)
    
    # Refill delay correlated with adherence
    refill_delay_days = []
    for adh in medication_adherence:
        if adh > 90:
            delay = np.random.randint(0, 3)
        elif adh > 70:
            delay = np.random.randint(1, 7)
        else:
            delay = np.random.randint(5, 18)
        refill_delay_days.append(delay)
    refill_delay_days = np.array(refill_delay_days)
    
    prescription_frequency = np.random.randint(1, 13, size=num_samples) # per year
    previous_anomalies = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05], size=num_samples)
    hospital_count = np.random.randint(1, 5, size=num_samples)
    
    # 2. Assign Risk level based on clinical scoring rules + minor normal noise
    risk_scores = []
    for i in range(num_samples):
        score = 0.0
        
        # Polypharmacy (lots of drugs)
        if num_medications[i] >= 8:
            score += 2.5
        elif num_medications[i] >= 5:
            score += 1.2
            
        # Age risk
        if patient_age[i] > 65:
            score += 2.0
        elif patient_age[i] > 50:
            score += 1.0
            
        # Adherence & Refill Delay
        score += ((100 - medication_adherence[i]) / 10.0) * 1.5
        score += refill_delay_days[i] * 0.3
        
        # Clinical indicators
        score += interaction_count[i] * 2.0
        score += allergy_flag[i] * 1.8
        score += chronic_disease_count[i] * 1.0
        
        # High dosage
        if dosage_amount[i] > 1000:
            score += 1.5
            
        risk_scores.append(score)
        
    risk_scores = np.array(risk_scores)
    noise = np.random.normal(0, 0.2, size=num_samples)
    final_scores = risk_scores + noise
    
    risk_levels = []
    for score in final_scores:
        if score < 7.5:
            risk_levels.append("Low")
        elif score < 13.0:
            risk_levels.append("Moderate")
        else:
            risk_levels.append("High")
            
    df = pd.DataFrame({
        "patient_age": patient_age,
        "gender": gender,
        "num_medications": num_medications,
        "dosage_amount": dosage_amount,
        "dosage_frequency": dosage_frequency,
        "treatment_duration": treatment_duration,
        "allergy_flag": allergy_flag,
        "chronic_disease_count": chronic_disease_count,
        "interaction_count": interaction_count,
        "medication_adherence": medication_adherence,
        "refill_delay_days": refill_delay_days,
        "prescription_frequency": prescription_frequency,
        "previous_anomalies": previous_anomalies,
        "hospital_count": hospital_count,
        "risk_level": risk_levels
    })
    
    df.to_csv(filename, index=False)
    print(f"Risk dataset created at '{filename}'. Shape: {df.shape}")
    return df

def train_risk():
    # 1. Generate / Load dataset
    df = generate_risk_dataset()
    print("Dataset loaded")
    
    # 2. Clean data
    df = df.dropna()
    
    # 3. Mathematical Feature Engineering (Z-scores)
    mean_dosage = float(df["dosage_amount"].mean())
    std_dosage = float(df["dosage_amount"].std())
    mean_frequency = float(df["prescription_frequency"].mean())
    std_frequency = float(df["prescription_frequency"].std())
    mean_medications = float(df["num_medications"].mean())
    
    # Apply formulas
    df["dosage_zscore"] = (df["dosage_amount"] - mean_dosage) / std_dosage
    df["prescription_frequency_zscore"] = (df["prescription_frequency"] - mean_frequency) / std_frequency
    
    # Save statistics dict as scaler.pkl
    scaler_dict = {
        "mean_dosage": mean_dosage,
        "std_dosage": std_dosage,
        "mean_frequency": mean_frequency,
        "std_frequency": std_frequency,
        "mean_medications": mean_medications
    }
    joblib.dump(scaler_dict, "ml/scaler.pkl")
    print("Features created")
    
    # 4. Prepare X and y
    # Encode gender
    df["gender_code"] = df["gender"].map({"M": 0, "F": 1})
    
    feature_cols = [
        "patient_age", "gender_code", "num_medications", "dosage_amount", 
        "dosage_frequency", "treatment_duration", "allergy_flag", "chronic_disease_count", 
        "interaction_count", "medication_adherence", "refill_delay_days", "prescription_frequency",
        "dosage_zscore", "prescription_frequency_zscore"
    ]
    
    X = df[feature_cols]
    y = df["risk_level"].map({"Low": 0, "Moderate": 1, "High": 2})
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Train & Compare Models
    print("Comparing Random Forest Classifier and Logistic Regression...")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    
    print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
    print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
    
    # Select the best
    if rf_acc >= lr_acc:
        best_model = rf_model
        best_preds = rf_preds
        model_name = "RandomForest"
        print("Selected model: RandomForestClassifier")
    else:
        best_model = lr_model
        best_preds = lr_preds
        model_name = "LogisticRegression"
        print("Selected model: LogisticRegression")
        
    print("Model trained")
    
    # 6. Evaluation metrics
    acc = accuracy_score(y_test, best_preds)
    prec = precision_score(y_test, best_preds, average="weighted")
    rec = recall_score(y_test, best_preds, average="weighted")
    f1 = f1_score(y_test, best_preds, average="weighted")
    
    # Feature Importances (for Random Forest)
    importances_dict = {}
    if model_name == "RandomForest":
        importances = best_model.feature_importances_
        for name, imp in zip(feature_cols, importances):
            importances_dict[name] = round(float(imp), 4)
        importances_dict = dict(sorted(importances_dict.items(), key=lambda x: x[1], reverse=True))
    else:
        # For LR, take absolute coefficients as proxy importance
        coefs = np.abs(best_model.coef_).mean(axis=0)
        coefs /= coefs.sum()
        for name, coef in zip(feature_cols, coefs):
            importances_dict[name] = round(float(coef), 4)
        importances_dict = dict(sorted(importances_dict.items(), key=lambda x: x[1], reverse=True))
        
    # Save the selected model
    joblib.dump(best_model, "ml/risk_model.pkl")
    
    # Save metrics JSON
    metrics = {
        "model_type": model_name,
        "accuracy": round(acc * 100, 1),
        "precision": round(prec * 100, 1),
        "recall": round(rec * 100, 1),
        "f1_score": round(f1 * 100, 1),
        "feature_importances": importances_dict
    }
    with open("ml/risk_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Model saved successfully")
    print(f"Final Accuracy: {acc * 100:.2f}% | F1: {f1 * 100:.2f}%")

if __name__ == "__main__":
    train_risk()
