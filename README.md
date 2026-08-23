# NALAM — Patient-Controlled Healthcare Data & Prescription Intelligence

**"Your Data. Your Consent. Smarter Healthcare."**

NALAM is a secure, patient-owned medical vault and clinical decision-support application built for the **MathX Innovation Hackathon**. NALAM empowers patients with granular control over their medical history, allergies, and active medications, and incorporates twin-engine Machine Learning (ML) models that perform real-time safety risk classification and mathematical prescription anomaly screening.

---

## 📖 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Proposed Solution](#-proposed-solution)
3. [System Architecture](#-system-architecture)
4. [Technology Stack](#-technology-stack)
5. [Mathematical Feature Engineering](#-mathematical-feature-engineering)
6. [Machine Learning Core](#-machine-learning-core)
7. [API Documentation](#-api-documentation)
8. [Installation & Setup](#-installation--setup)
9. [Hackathon Demo Scenario](#-hackathon-demo-scenario)
10. [Limitations & Future Scope](#-limitations--future-scope)

---

## 🔍 Problem Statement
Healthcare records are fragmented across multiple clinics and paper prescription slips. Patients suffer from:
* Scattered health records leading to medical history gaps.
* Lack of granular control over who can inspect private records.
* Accidental adverse drug events due to overlapping prescriptions and drug-drug interactions.
* Unsupervised polypharmacy (taking too many medications concurrently) and prescription anomalies.

---

## 💡 Proposed Solution
NALAM acts as an intelligent medical record custodian:
1. **Patient-Controlled Medical Vault**: Secure storage of clinical variables (allergies, conditions, historical prescriptions) where data sharing is restricted by default.
2. **Granular Consent Workflow**: Doctors must request access to specific categories (e.g. only allergies, not insurance), and patients authorize access via a secure OTP validation.
3. **Twin-Engine ML Predictors**: 
   * **Supervised Risk Classifier**: Assesses medication risk index (Low / Moderate / High) based on patient attributes.
   * **Unsupervised Anomaly Detector**: Flags unusual statistical patterns (e.g. outlier dosages or refill delays) relative to historical baselines.
4. **Emergency Summary Mode**: Immediate clinical dashboard for first responders displaying blood type and critical allergies during emergency events.

---

## 🏗 System Architecture

```text
                     NALAM
                       │
                       ▼
                PATIENT PORTAL
                       │
                       ▼
                SECURE DATA VAULT
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
     PRESCRIPTIONS            MEDICAL DATA
           │                       │
           └───────────┬───────────┘
                       ▼
                CONSENT LAYER
                       │
                       ▼
              AUTHENTICATION / OTP
                       │
                       ▼
                DATA PROCESSING
                       │
                       ▼
             FEATURE ENGINEERING
                       │
                       ▼
            MATHEMATICAL FEATURES
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Mean         Z-score     Adherence
          │            │            │
          └────────────┼────────────┘
                       ▼
                 ML ENGINE
                       │
              ┌────────┴────────┐
              ▼                 ▼
     LOGISTIC REGRESSION  ISOLATION FOREST
       RISK MODEL         ANOMALY MODEL
              │                 │
              ▼                 ▼
        Risk Level          Anomaly
              │                 │
              └────────┬────────┘
                       ▼
               EXPLAINABLE INSIGHT
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       PATIENT       DOCTOR       ADMIN
        ALERTS        VIEW       ANALYTICS
```

---

## 🛠 Technology Stack
* **Frontend**: HTML5, CSS3 (Teal/Emerald light clinical theme), JavaScript (Vanilla ES6), Chart.js (Interactive feature weights visualization).
* **Backend**: Python 3, Flask.
* **Database**: SQLite (SQL schema mapping clinical relationships).
* **Machine Learning**: Pandas, NumPy, Scikit-learn, Joblib.

---

## 📐 Mathematical Feature Engineering

### A. Medication Adherence Rate
Matches actual drug intake compliance:
$$\text{Adherence (\%)} = \frac{\text{Doses Taken}}{\text{Doses Expected}} \times 100$$

### B. Mean & Standard Deviation
Establishes historical baseline averages ($\mu$) and standard deviations ($\sigma$) across global dataset distributions for clinical metrics (Medication Counts, Dosages, Refill Delays).

### C. Z-Score Normalization
Transforms raw dosage ($x_D$) and prescription frequencies ($x_F$) into standard score deviations to evaluate relative anomalies:
$$Z_D = \frac{x_D - \mu_D}{\sigma_D}$$
$$Z_F = \frac{x_F - \mu_F}{\sigma_F}$$

### D. Drug Interaction Risk Score
Calculates compounding drug risks:
$$\text{Interaction Score} = \sum (\text{Interaction Indicator} \times \text{Severity Score})$$
Mock interaction indices:
* `Warfarin` + `Aspirin` = Severity `3.0` (High bleeding risk)
* `Sildenafil` + `Nitroglycerin` = Severity `5.0` (Critical hypotension risk)

---

## 🧠 Machine Learning Core

### 1. Risk Classification Model
* **Algorithm**: Evaluated and compared `RandomForestClassifier` and `LogisticRegression`.
* **Selection**: `LogisticRegression` achieved **97.1% accuracy** and **97.1% F1 Score** on stratified test split.
* **Saved Artifacts**: `ml/risk_model.pkl`, `ml/scaler.pkl`, `ml/risk_metrics.json`.

### 2. Prescription Anomaly Detection
* **Algorithm**: Unsupervised `IsolationForest` detecting outlier distributions.
* **Selection**: Achieved **99.3% accuracy** in distinguishing normal prescribing behaviors from anomalous clusters.
* **Saved Artifacts**: `ml/anomaly_model.pkl`, `ml/anomaly_metrics.json`.

---

## 🔌 API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/login` | POST | Authenticate Patient/Doctor. |
| `/api/predict-risk` | POST | Returns risk level (`Low`/`Moderate`/`High`) and explainable signals. |
| `/api/detect-anomaly` | POST | Returns anomaly status (`Normal`/`Anomaly`) and decision metrics. |
| `/api/refill-prediction` | POST | Computes estimated remaining days and warning levels. |
| `/api/consent/request` | POST | Doctor requests access to patient profile. |
| `/api/consent/approve` | POST | Patient approves doctor access request and generates OTP. |
| `/api/doctor/verify-otp` | POST | Doctor enters authorization OTP to unlock records. |
| `/api/doctor/view-patient` | POST | Fetch consented medical records (restricted by flags). |
| `/api/consent/revoke` | POST | Patient terminates doctor session immediately. |
| `/api/chat` | POST | Rule-based chatbot querying secure vault variables. |
| `/api/emergency/<id>` | GET | Fast bypass portal returning blood group and active allergies. |

---

## 🚀 Installation & Setup

1. **Clone & Navigate**:
   ```bash
   cd NALAM
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Initialize SQLite Database**:
   ```bash
   python database/init_db.py
   ```
4. **Train ML Models**:
   ```bash
   python ml/train_risk_model.py
   python ml/train_anomaly_model.py
   ```
5. **Start Flask Server**:
   ```bash
   python app.py
   ```
6. **Browse**: Open `http://127.0.0.1:5000` in your web browser.

---

## 🎬 Hackathon Demo Scenario

### Demo Accounts:
* **Patient**: `patient@example.com` (password: `patient123`)
* **Doctor**: `doctor@example.com` (password: `doctor123`)

### Steps:
1. **Patient Login**: Click `Log in as Patient (John Doe)`. Review the dashboard: Metformin and Lisinopril refills, allergies (Penicillin), and prescription history.
2. **Prescription Risk Check**: Select the pre-seeded Metformin prescription and click `Analyze Safety`. The ML engine predicts **Low Risk** with high probability.
3. **Add Anomaly Prescription**: Click `Add Prescription` -> Click `Load Anomalous Demo`. This pre-fills Nitroglycerin at 5000mg with 45% adherence and 5 interactions. Click `Analyze Prescription`. The twin models predict **High Risk** (with 100% confidence) and flag a **Prescription Anomaly** (Isolation Forest anomaly score $<0$).
4. **Doctor Login**: Log out, click `Log in as Doctor (Dr. Smith)`. Search for `patient@example.com`. Notice access is **Restricted**. Click `Request Record Access`.
5. **Patient Approval**: Log out, log back in as Patient. In the *Consent Access Manager*, check "Medical History" and "Allergies" but uncheck "Current Medications". Click `Approve Access`. A **dynamic OTP** (e.g. `847192`) is shown.
6. **OTP Verification**: Log back in as Doctor. Enter the OTP code. Click `Verify`. Access is approved! Click `Load Unlocked Records`. Notice you can see John Doe's allergies and history, but his active medications are hidden (granular consent).
7. **Clinical Addition**: Fill the Prescription Console to prescribe a new medicine. Submit it.
8. **Revocation**: Log back in as Patient. Click `Revoke Access`. Doctor Sarah Smith can no longer access the record.

---

## ⚠️ Limitations & Future Scope
* **Limitations**: Trained on synthetic data. Models are prototypes intended for clinical decision support and do not diagnose disease.
* **Future Scope**:
  * Consent-based interoperability (FHIR / HL7 standard integration).
  * OCR engine to digitize handwritten prescriptions.
  * Blockchain Ledger to secure patient consent audits.
  * Real OTP transmission via SMS.
