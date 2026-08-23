import json
import os

def evaluate():
    # Load metrics
    try:
        risk_path = "ml/risk_metrics.json"
        anomaly_path = "ml/anomaly_metrics.json"
        
        if not os.path.exists(risk_path) or not os.path.exists(anomaly_path):
            print("Error: Models have not been trained yet. Please run training scripts.")
            return
            
        with open(risk_path, "r") as f:
            risk = json.load(f)
        with open(anomaly_path, "r") as f:
            anomaly = json.load(f)
            
        print("\n=== NALAM MODEL EVALUATION REPORT ===")
        print(f"Risk Model Type: {risk['model_type']}")
        print(f"Risk Model Accuracy: {risk['accuracy']}%")
        print(f"Risk Model Precision: {risk['precision']}%")
        print(f"Risk Model Recall: {risk['recall']}%")
        print(f"Risk Model F1 Score: {risk['f1_score']}%")
        
        print("\nTop Feature Importances (Risk Classifier):")
        for i, (feat, val) in enumerate(risk['feature_importances'].items()):
            if i < 6:
                print(f"  {i+1}. {feat:30} : {val * 100:.2f}%")
                
        print("\nAnomaly Model Type: IsolationForest")
        print(f"Anomaly Detection Accuracy: {anomaly['accuracy']}%")
        print(f"Total Anomalies Flagged: {anomaly['total_anomalies_flagged']}")
        print("=====================================\n")
    except Exception as e:
        print(f"Evaluation error: {str(e)}")

if __name__ == "__main__":
    evaluate()
