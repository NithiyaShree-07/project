import os
import json
import sqlite3
import random
from datetime import datetime, timedelta
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from feature_engineering import calculate_interaction_score, compute_zscore, calculate_adherence

app = Flask(__name__)
app.secret_key = "nalam_secret_key_for_session_security"

# Config uploads
UPLOAD_FOLDER = os.path.join("uploads", "prescriptions")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load ML models and scaler
risk_model = None
anomaly_model = None
scaler = None
risk_metrics = None
anomaly_metrics = None

def load_ml_assets():
    global risk_model, anomaly_model, scaler, risk_metrics, anomaly_metrics
    try:
        if os.path.exists("ml/risk_model.pkl"):
            risk_model = joblib.load("ml/risk_model.pkl")
        if os.path.exists("ml/anomaly_model.pkl"):
            anomaly_model = joblib.load("ml/anomaly_model.pkl")
        if os.path.exists("ml/scaler.pkl"):
            scaler = joblib.load("ml/scaler.pkl")
        if os.path.exists("ml/risk_metrics.json"):
            with open("ml/risk_metrics.json", "r") as f:
                risk_metrics = json.load(f)
        if os.path.exists("ml/anomaly_metrics.json"):
            with open("ml/anomaly_metrics.json", "r") as f:
                anomaly_metrics = json.load(f)
        print("ML assets loaded successfully.")
    except Exception as e:
        print(f"Error loading ML assets: {str(e)}")

# Try loading ML assets at startup
load_ml_assets()

# Database helper
def get_db():
    conn = sqlite3.connect(os.path.join("database", "database.db"))
    conn.row_factory = sqlite3.Row
    return conn

# Session patient helper
def get_current_user_profile(db, user_id, role):
    if role == "Patient":
        cursor = db.execute("SELECT * FROM patients WHERE user_id = ?", (user_id,))
        return cursor.fetchone()
    elif role == "Doctor":
        cursor = db.execute("SELECT * FROM doctors WHERE user_id = ?", (user_id,))
        return cursor.fetchone()
    return None

@app.route("/")
def index():
    if "user_id" not in session:
        return render_template("index.html", user=None, profile=None)
        
    db = get_db()
    profile = get_current_user_profile(db, session["user_id"], session["role"])
    db.close()
    
    # Reload models dynamically if they were not loaded yet
    if risk_model is None or anomaly_model is None:
        load_ml_assets()
        
    return render_template("index.html", user=session, profile=profile)

# --- Authentication APIs ---

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.json or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")  # Patient or Doctor
    
    if not username or not email or not password or not role:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
    if role not in ["Patient", "Doctor"]:
        return jsonify({"status": "error", "message": "Invalid user role"}), 400
        
    db = get_db()
    try:
        password_hash = generate_password_hash(password)
        cursor = db.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, role)
        )
        user_id = cursor.lastrowid
        
        # Initialize profiles
        if role == "Patient":
            age = int(data.get("age", 30))
            gender = data.get("gender", "M")
            blood_group = data.get("blood_group", "O+")
            name = data.get("name", username)
            
            db.execute(
                "INSERT INTO patients (user_id, name, age, gender, blood_group) VALUES (?, ?, ?, ?, ?)",
                (user_id, name, age, gender, blood_group)
            )
        elif role == "Doctor":
            specialty = data.get("specialty", "General Practice")
            hospital = data.get("hospital", "Clinic")
            name = data.get("name", username)
            
            db.execute(
                "INSERT INTO doctors (user_id, name, specialty, hospital) VALUES (?, ?, ?, ?)",
                (user_id, name, specialty, hospital)
            )
            
        db.commit()
        return jsonify({"status": "success", "message": "Registration successful. Please log in."})
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "Email is already registered"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    
    if not email or not password:
        return jsonify({"status": "error", "message": "Missing email or password"}), 400
        
    db = get_db()
    cursor = db.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if not user or not check_password_hash(user["password_hash"], password):
        db.close()
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401
        
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["email"] = user["email"]
    session["role"] = user["role"]
    
    profile = get_current_user_profile(db, user["id"], user["role"])
    session["profile_id"] = profile["id"]
    
    db.close()
    return jsonify({
        "status": "success", 
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "profile_id": profile["id"]
        }
    })

@app.route("/api/logout", methods=["GET", "POST"])
def api_logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out successfully"})

# --- Patient Vault Records APIs ---

@app.route("/api/patient/records", methods=["GET"])
def get_patient_records():
    if "user_id" not in session or session["role"] != "Patient":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    patient_id = session["profile_id"]
    db = get_db()
    
    # Fetch Allergies
    allergies = [row["allergy_name"] for row in db.execute("SELECT allergy_name FROM allergies WHERE patient_id = ?", (patient_id,)).fetchall()]
    
    # Fetch Chronic Diseases
    chronic = [row["disease_name"] for row in db.execute("SELECT disease_name FROM chronic_diseases WHERE patient_id = ?", (patient_id,)).fetchall()]
    
    # Fetch Medications & Refill status
    meds = db.execute("SELECT * FROM refill_records WHERE patient_id = ?", (patient_id,)).fetchall()
    meds_list = [dict(row) for row in meds]
    
    # Fetch Prescription History
    rx = db.execute("SELECT * FROM prescriptions WHERE patient_id = ? ORDER BY id DESC", (patient_id,)).fetchall()
    rx_list = []
    for r in rx:
        items = db.execute("SELECT * FROM prescription_items WHERE prescription_id = ?", (r["id"],)).fetchall()
        rx_list.append({
            "id": r["id"],
            "doctor_name": r["doctor_name"],
            "hospital_name": r["hospital_name"],
            "prescription_date": r["prescription_date"],
            "uploaded_file": r["uploaded_file"],
            "items": [dict(item) for item in items]
        })
        
    db.close()
    return jsonify({
        "status": "success",
        "allergies": allergies,
        "chronic_diseases": chronic,
        "medications": meds_list,
        "prescriptions": rx_list
    })

# --- ML & Mathematics Predictions APIs ---

@app.route("/api/predict-risk", methods=["POST"])
def predict_risk():
    global risk_model, scaler
    if risk_model is None or scaler is None:
        load_ml_assets()
        if risk_model is None or scaler is None:
            return jsonify({"status": "error", "message": "ML assets are not loaded"}), 500
            
    try:
        data = request.json or {}
        
        # 1. Capture inputs
        age = float(data.get("age", 30))
        gender_code = float(data.get("gender_code", 0)) # M=0, F=1
        num_medications = float(data.get("num_medications", 1))
        dosage_amount = float(data.get("dosage_amount", 500))
        dosage_frequency = float(data.get("dosage_frequency", 1))
        treatment_duration = float(data.get("treatment_duration", 30))
        allergy_flag = float(data.get("allergy_flag", 0))
        chronic_disease_count = float(data.get("chronic_disease_count", 0))
        interaction_count = float(data.get("interaction_count", 0))
        medication_adherence = float(data.get("medication_adherence", 100))
        refill_delay_days = float(data.get("refill_delay_days", 0))
        prescription_frequency = float(data.get("prescription_frequency", 1))
        
        # 2. Perform Mathematical Feature Engineering (Z-scores)
        mean_dosage = scaler.get("mean_dosage", 500.0)
        std_dosage = scaler.get("std_dosage", 300.0)
        mean_frequency = scaler.get("mean_frequency", 6.5)
        std_frequency = scaler.get("std_frequency", 3.5)
        
        dosage_zscore = compute_zscore(dosage_amount, mean_dosage, std_dosage)
        prescription_frequency_zscore = compute_zscore(prescription_frequency, mean_frequency, std_frequency)
        
        # 3. Create feature array matching RandomForest column list
        features = np.array([[
            age, gender_code, num_medications, dosage_amount, dosage_frequency, 
            treatment_duration, allergy_flag, chronic_disease_count, interaction_count, 
            medication_adherence, refill_delay_days, prescription_frequency,
            dosage_zscore, prescription_frequency_zscore
        ]])
        
        # 4. Predict
        prediction_idx = int(risk_model.predict(features)[0]) # 0=Low, 1=Moderate, 2=High
        probabilities = risk_model.predict_proba(features)[0]
        
        labels_map = {0: "Low", 1: "Moderate", 2: "High"}
        risk_level = labels_map[prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        # Explainable factors
        explainable_signals = []
        if interaction_count >= 3:
            explainable_signals.append("🔴 High drug-to-drug interaction count")
        if medication_adherence < 70:
            explainable_signals.append("🔴 Low medication adherence")
        if refill_delay_days > 7:
            explainable_signals.append("🟠 Elevated refill delay")
        if num_medications >= 8:
            explainable_signals.append("🟡 Polypharmacy (multiple active medications)")
        if abs(dosage_zscore) > 1.5:
            explainable_signals.append(f"🟡 Outlier dosage deviation (Z-score: {dosage_zscore:.2f})")
            
        if not explainable_signals:
            explainable_signals.append("🟢 Standard prescribing metrics")
            
        # Recommendations
        if risk_level == "High":
            msg = "Potential medication risk detected. Please consult a qualified healthcare professional."
        elif risk_level == "Moderate":
            msg = "Moderate risk indicators observed. Professional verification recommended."
        else:
            msg = "Low risk profile. Standard prescription guidelines apply."
            
        return jsonify({
            "status": "success",
            "risk_level": risk_level,
            "confidence": round(confidence * 100, 1),
            "message": msg,
            "dosage_zscore": round(dosage_zscore, 2),
            "prescription_frequency_zscore": round(prescription_frequency_zscore, 2),
            "contributing_signals": explainable_signals
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/detect-anomaly", methods=["POST"])
def detect_anomaly():
    global anomaly_model, scaler
    if anomaly_model is None or scaler is None:
        load_ml_assets()
        if anomaly_model is None:
            return jsonify({"status": "error", "message": "Anomaly model is not loaded"}), 500
            
    try:
        data = request.json or {}
        num_medications = float(data.get("num_medications", 1))
        dosage_amount = float(data.get("dosage_amount", 500))
        prescription_frequency = float(data.get("prescription_frequency", 1))
        refill_delay_days = float(data.get("refill_delay_days", 0))
        interaction_count = float(data.get("interaction_count", 0))
        
        # Scaling parameters
        mean_dosage = scaler.get("mean_dosage", 500.0)
        std_dosage = scaler.get("std_dosage", 300.0)
        mean_frequency = scaler.get("mean_frequency", 6.5)
        std_frequency = scaler.get("std_frequency", 3.5)
        mean_medications = scaler.get("mean_medications", 5.0)
        std_medications = 2.5
        
        dosage_zscore = compute_zscore(dosage_amount, mean_dosage, std_dosage)
        prescription_frequency_zscore = compute_zscore(prescription_frequency, mean_frequency, std_frequency)
        medicine_count_zscore = compute_zscore(num_medications, mean_medications, std_medications)
        
        # Feature columns matching train_anomaly_model:
        # num_medications, dosage_amount, prescription_frequency, refill_delay_days, 
        # interaction_count, dosage_zscore, prescription_frequency_zscore, medicine_count_zscore
        features = np.array([[
            num_medications, dosage_amount, prescription_frequency, refill_delay_days,
            interaction_count, dosage_zscore, prescription_frequency_zscore, medicine_count_zscore
        ]])
        
        prediction = int(anomaly_model.predict(features)[0]) # -1 = Anomaly, 1 = Normal
        anomaly_score = float(anomaly_model.decision_function(features)[0])
        
        is_anomaly = (prediction == -1)
        
        if is_anomaly:
            msg = "Potential prescription anomaly detected. Expert verification recommended."
        else:
            msg = "Statistical patterns align with standard prescribing guidelines."
            
        return jsonify({
            "status": "success",
            "anomaly_status": "Anomaly" if is_anomaly else "Normal",
            "anomaly_score": round(anomaly_score, 3),
            "message": msg,
            "z_scores": {
                "dosage_zscore": round(dosage_zscore, 2),
                "prescription_frequency_zscore": round(prescription_frequency_zscore, 2),
                "medicine_count_zscore": round(medicine_count_zscore, 2)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/refill-prediction", methods=["POST"])
def refill_prediction():
    try:
        data = request.json or {}
        remaining_quantity = float(data.get("remaining_quantity", 0))
        daily_quantity = float(data.get("daily_quantity", 1))
        
        if daily_quantity <= 0:
            return jsonify({"status": "error", "message": "Daily quantity must be greater than zero."}), 400
            
        days_remaining = int(remaining_quantity / daily_quantity)
        
        # Reminder level mapping
        if days_remaining > 7:
            level = "Normal"
            alert_class = "success"
        elif 4 <= days_remaining <= 7:
            level = "Reminder"
            alert_class = "warning"
        elif 1 <= days_remaining <= 3:
            level = "Urgent reminder"
            alert_class = "danger"
        else:
            level = "Refill required"
            alert_class = "danger"
            
        return jsonify({
            "status": "success",
            "days_remaining": days_remaining,
            "reminder_level": level,
            "alert_class": alert_class
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Doctor Portal & Consent workflow APIs ---

@app.route("/api/doctor/search", methods=["POST"])
def doctor_search_patient():
    if "user_id" not in session or session["role"] != "Doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    email = data.get("email", "").strip()
    
    db = get_db()
    cursor = db.execute("""
        SELECT p.*, u.email FROM patients p 
        JOIN users u ON p.user_id = u.id 
        WHERE u.email = ?
    """, (email,))
    patient = cursor.fetchone()
    
    if not patient:
        db.close()
        return jsonify({"status": "error", "message": "Patient not found"}), 404
        
    # Check if there is an active consent request
    doctor_id = session["profile_id"]
    cursor = db.execute("""
        SELECT * FROM consent_requests 
        WHERE patient_id = ? AND doctor_id = ? 
        ORDER BY id DESC LIMIT 1
    """, (patient["id"], doctor_id))
    consent = cursor.fetchone()
    
    consent_status = "None"
    share_history = 0
    share_meds = 0
    share_allergies = 0
    
    if consent:
        consent_status = consent["status"]
        # Check if consent is active and not expired
        if consent_status == "Approved" and consent["expiry_at"]:
            expiry = datetime.strptime(consent["expiry_at"], "%Y-%m-%d %H:%M:%S")
            if datetime.now() > expiry:
                # Consent expired, update in DB
                db.execute("UPDATE consent_requests SET status = 'Revoked' WHERE id = ?", (consent["id"],))
                db.commit()
                consent_status = "Revoked"
                
        if consent_status == "Approved":
            share_history = consent["share_medical_history"]
            share_meds = consent["share_medications"]
            share_allergies = consent["share_allergies"]
            
    # Log lookup audit
    db.close()
    return jsonify({
        "status": "success",
        "patient": {
            "id": patient["id"],
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"],
            "blood_group": patient["blood_group"] if share_history else "🔒 Restricted",
            "email": patient["email"]
        },
        "consent": {
            "status": consent_status,
            "share_medical_history": share_history,
            "share_medications": share_meds,
            "share_allergies": share_allergies
        }
    })

@app.route("/api/consent/request", methods=["POST"])
def request_consent():
    if "user_id" not in session or session["role"] != "Doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    patient_id = data.get("patient_id")
    
    if not patient_id:
        return jsonify({"status": "error", "message": "Missing patient ID"}), 400
        
    doctor_id = session["profile_id"]
    otp = str(random.randint(100000, 999999))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    db = get_db()
    # Check if there is already a pending request, update or insert
    cursor = db.execute("""
        SELECT id FROM consent_requests 
        WHERE patient_id = ? AND doctor_id = ? AND status = 'Pending'
    """, (patient_id, doctor_id))
    pending = cursor.fetchone()
    
    if pending:
        db.execute("""
            UPDATE consent_requests 
            SET otp = ?, created_at = ? 
            WHERE id = ?
        """, (otp, now_str, pending["id"]))
    else:
        db.execute("""
            INSERT INTO consent_requests (patient_id, doctor_id, status, otp, otp_verified, created_at)
            VALUES (?, ?, 'Pending', ?, 0, ?)
        """, (patient_id, doctor_id, otp, now_str))
        
    # Audit log
    db.execute("""
        INSERT INTO access_logs (doctor_id, patient_id, access_type, timestamp, details)
        VALUES (?, ?, 'Requested', ?, 'Doctor requested data access')
    """, (doctor_id, patient_id, now_str))
    
    db.commit()
    db.close()
    
    return jsonify({
        "status": "success", 
        "message": "Consent request sent. Patient must verify the authorization OTP."
    })

@app.route("/api/patient/consent-requests", methods=["GET"])
def get_patient_consent_requests():
    if "user_id" not in session or session["role"] != "Patient":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    patient_id = session["profile_id"]
    db = get_db()
    cursor = db.execute("""
        SELECT r.*, d.name as doctor_name, d.specialty, d.hospital 
        FROM consent_requests r 
        JOIN doctors d ON r.doctor_id = d.id 
        WHERE r.patient_id = ? AND r.status = 'Pending'
    """, (patient_id,))
    requests_list = [dict(row) for row in cursor.fetchall()]
    db.close()
    return jsonify({"status": "success", "requests": requests_list})

@app.route("/api/consent/approve", methods=["POST"])
def approve_consent():
    if "user_id" not in session or session["role"] != "Patient":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    request_id = data.get("request_id")
    share_history = int(data.get("share_history", 0))
    share_meds = int(data.get("share_meds", 0))
    share_allergies = int(data.get("share_allergies", 0))
    
    if not request_id:
        return jsonify({"status": "error", "message": "Missing request ID"}), 400
        
    db = get_db()
    cursor = db.execute("SELECT * FROM consent_requests WHERE id = ? AND patient_id = ?", (request_id, session["profile_id"]))
    req = cursor.fetchone()
    
    if not req:
        db.close()
        return jsonify({"status": "error", "message": "Consent request not found"}), 404
        
    # Generate expiry (e.g. 15 minutes for security)
    expiry = (datetime.now() + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    db.execute("""
        UPDATE consent_requests 
        SET status = 'Approved', share_medical_history = ?, share_medications = ?, share_allergies = ?, expiry_at = ?
        WHERE id = ?
    """, (share_history, share_meds, share_allergies, expiry, request_id))
    
    # Audit log
    db.execute("""
        INSERT INTO access_logs (doctor_id, patient_id, access_type, timestamp, details)
        VALUES (?, ?, 'Approved', ?, ?)
    """, (req["doctor_id"], req["patient_id"], now_str, f"Patient approved request. Granular access granted. Expiry: {expiry}"))
    
    db.commit()
    db.close()
    
    return jsonify({
        "status": "success",
        "message": "Consent approved successfully. Access code generated.",
        "otp": req["otp"]
    })

@app.route("/api/doctor/verify-otp", methods=["POST"])
def doctor_verify_otp():
    if "user_id" not in session or session["role"] != "Doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    patient_id = data.get("patient_id")
    otp = data.get("otp", "").strip()
    
    if not patient_id or not otp:
        return jsonify({"status": "error", "message": "Missing patient ID or OTP"}), 400
        
    doctor_id = session["profile_id"]
    db = get_db()
    
    # Query approved request
    cursor = db.execute("""
        SELECT * FROM consent_requests 
        WHERE patient_id = ? AND doctor_id = ? AND otp = ? AND status = 'Approved'
    """, (patient_id, doctor_id, otp))
    request_row = cursor.fetchone()
    
    if not request_row:
        db.close()
        return jsonify({"status": "error", "message": "Invalid OTP code"}), 400
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("UPDATE consent_requests SET otp_verified = 1 WHERE id = ?", (request_row["id"],))
    
    # Audit log
    db.execute("""
        INSERT INTO access_logs (doctor_id, patient_id, access_type, timestamp, details)
        VALUES (?, ?, 'OTP Verified', ?, 'Doctor completed OTP verification')
    """, (doctor_id, patient_id, now_str))
    
    db.commit()
    db.close()
    
    return jsonify({"status": "success", "message": "Access authorized successfully!"})

@app.route("/api/consent/revoke", methods=["POST"])
def revoke_consent():
    if "user_id" not in session or session["role"] != "Patient":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    doctor_id = data.get("doctor_id")
    
    if not doctor_id:
        return jsonify({"status": "error", "message": "Missing doctor ID"}), 400
        
    patient_id = session["profile_id"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    db = get_db()
    db.execute("""
        UPDATE consent_requests 
        SET status = 'Revoked' 
        WHERE patient_id = ? AND doctor_id = ? AND status = 'Approved'
    """, (patient_id, doctor_id))
    
    # Audit log
    db.execute("""
        INSERT INTO access_logs (doctor_id, patient_id, access_type, timestamp, details)
        VALUES (?, ?, 'Revoked', ?, 'Patient revoked doctor data access')
    """, (doctor_id, patient_id, now_str))
    
    db.commit()
    db.close()
    
    return jsonify({"status": "success", "message": "Doctor access privileges revoked."})

@app.route("/api/patient/consent-history", methods=["GET"])
def get_patient_consent_history():
    if "user_id" not in session or session["role"] != "Patient":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    patient_id = session["profile_id"]
    db = get_db()
    
    # Active consents
    cursor1 = db.execute("""
        SELECT r.*, d.name as doctor_name, d.specialty, d.hospital 
        FROM consent_requests r 
        JOIN doctors d ON r.doctor_id = d.id 
        WHERE r.patient_id = ? AND r.status = 'Approved'
    """, (patient_id,))
    active = [dict(row) for row in cursor1.fetchall()]
    
    # Audit log logs
    cursor2 = db.execute("""
        SELECT l.*, d.name as doctor_name 
        FROM access_logs l 
        JOIN doctors d ON l.doctor_id = d.id 
        WHERE l.patient_id = ? ORDER BY l.id DESC LIMIT 15
    """, (patient_id,))
    logs = [dict(row) for row in cursor2.fetchall()]
    
    db.close()
    return jsonify({
        "status": "success",
        "active_consents": active,
        "logs": logs
    })

@app.route("/api/doctor/view-patient", methods=["POST"])
def doctor_view_patient():
    if "user_id" not in session or session["role"] != "Doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    patient_id = data.get("patient_id")
    
    if not patient_id:
        return jsonify({"status": "error", "message": "Missing patient ID"}), 400
        
    doctor_id = session["profile_id"]
    db = get_db()
    
    # Validate consent
    cursor = db.execute("""
        SELECT * FROM consent_requests 
        WHERE patient_id = ? AND doctor_id = ? AND status = 'Approved' AND otp_verified = 1
        ORDER BY id DESC LIMIT 1
    """, (patient_id, doctor_id))
    consent = cursor.fetchone()
    
    if not consent:
        db.close()
        return jsonify({"status": "error", "message": "Access not approved or OTP not verified."}), 403
        
    # Check expiry
    expiry = datetime.strptime(consent["expiry_at"], "%Y-%m-%d %H:%M:%S")
    if datetime.now() > expiry:
        db.execute("UPDATE consent_requests SET status = 'Revoked' WHERE id = ?", (consent["id"],))
        db.commit()
        db.close()
        return jsonify({"status": "error", "message": "Consent session has expired."}), 403
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Audit Log
    db.execute("""
        INSERT INTO access_logs (doctor_id, patient_id, access_type, timestamp, details)
        VALUES (?, ?, 'Record Accessed', ?, 'Doctor accessed patient medical file')
    """, (doctor_id, patient_id, now_str))
    
    # Load patient demographics
    p = db.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    
    # Fetch conditional data
    response = {
        "status": "success",
        "patient": {
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "blood_group": p["blood_group"] if consent["share_medical_history"] else "🔒 Shared Consent Required"
        },
        "allergies": [row["allergy_name"] for row in db.execute("SELECT allergy_name FROM allergies WHERE patient_id = ?", (patient_id,)).fetchall()] if consent["share_allergies"] else None,
        "chronic_diseases": [row["disease_name"] for row in db.execute("SELECT disease_name FROM chronic_diseases WHERE patient_id = ?", (patient_id,)).fetchall()] if consent["share_medical_history"] else None,
    }
    
    if consent["share_medications"]:
        meds = db.execute("SELECT * FROM refill_records WHERE patient_id = ?", (patient_id,)).fetchall()
        response["medications"] = [dict(row) for row in meds]
        
        rx = db.execute("SELECT * FROM prescriptions WHERE patient_id = ? ORDER BY id DESC", (patient_id,)).fetchall()
        rx_list = []
        for r in rx:
            items = db.execute("SELECT * FROM prescription_items WHERE prescription_id = ?", (r["id"],)).fetchall()
            rx_list.append({
                "id": r["id"],
                "doctor_name": r["doctor_name"],
                "hospital_name": r["hospital_name"],
                "prescription_date": r["prescription_date"],
                "items": [dict(item) for item in items]
            })
        response["prescriptions"] = rx_list
        
    # Calculate access remaining time
    seconds_left = max(0, int((expiry - datetime.now()).total_seconds()))
    response["seconds_left"] = seconds_left
    
    db.commit()
    db.close()
    return jsonify(response)

@app.route("/api/doctor/prescription", methods=["POST"])
def add_prescription_by_doctor():
    if "user_id" not in session or session["role"] != "Doctor":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    patient_id = data.get("patient_id")
    medicine_name = data.get("medicine_name")
    dosage_amount = int(data.get("dosage_amount", 500))
    dosage_frequency = int(data.get("dosage_frequency", 1))
    treatment_duration = int(data.get("treatment_duration", 30))
    refill_delay_days = int(data.get("refill_delay_days", 1))
    prescription_frequency = int(data.get("prescription_frequency", 6))
    
    if not patient_id or not medicine_name:
        return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
    doctor_id = session["profile_id"]
    db = get_db()
    
    # Verify active consent
    consent = db.execute("""
        SELECT * FROM consent_requests 
        WHERE patient_id = ? AND doctor_id = ? AND status = 'Approved' AND otp_verified = 1
        ORDER BY id DESC LIMIT 1
    """, (patient_id, doctor_id)).fetchone()
    
    if not consent:
        db.close()
        return jsonify({"status": "error", "message": "Write permission denied: Consent not active"}), 403
        
    d = db.execute("SELECT name, hospital FROM doctors WHERE id = ?", (doctor_id,)).fetchone()
    now_str = datetime.now().strftime("%Y-%m-%d")
    
    # Calculate interactive score
    # Fetch active medicines to scan interaction
    active_meds = [row["medicine_name"] for row in db.execute("SELECT medicine_name FROM refill_records WHERE patient_id = ?", (patient_id,)).fetchall()]
    active_meds.append(medicine_name)
    interaction_score = calculate_interaction_score(active_meds)
    
    # Fetch history to compute adherence percentage (defaulting to 90 for demo)
    adherence = 90
    
    # 1. Insert Prescription
    cursor = db.execute("""
        INSERT INTO prescriptions (patient_id, doctor_name, hospital_name, prescription_date)
        VALUES (?, ?, ?, ?)
    """, (patient_id, d["name"], d["hospital"], now_str))
    rx_id = cursor.lastrowid
    
    # 2. Insert Item
    db.execute("""
        INSERT INTO prescription_items (prescription_id, medicine_name, dosage_amount, dosage_frequency, treatment_duration, refill_delay_days, prescription_frequency, interaction_count, adherence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (rx_id, medicine_name, dosage_amount, dosage_frequency, treatment_duration, refill_delay_days, prescription_frequency, int(interaction_score), adherence))
    
    # 3. Add to refill records
    refill_date = (datetime.now() + timedelta(days=treatment_duration)).strftime("%Y-%m-%d")
    db.execute("""
        INSERT INTO refill_records (patient_id, medicine_name, remaining_quantity, daily_quantity, estimated_refill_date)
        VALUES (?, ?, ?, ?, ?)
    """, (patient_id, medicine_name, dosage_frequency * treatment_duration, dosage_frequency, refill_date))
    
    db.commit()
    db.close()
    return jsonify({"status": "success", "message": "Prescription successfully generated & logged in Patient Vault!"})

# --- Local Chat Assistant Endpoint ---

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if "user_id" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    data = request.json or {}
    message = data.get("message", "").lower().strip()
    role = session["role"]
    
    # Simple Local Rule-Based Chat Assistant utilizing Database context
    db = get_db()
    reply = ""
    
    if role == "Patient":
        patient_id = session["profile_id"]
        
        # Read patient's active clinical variables for context
        allergies = [row["allergy_name"] for row in db.execute("SELECT allergy_name FROM allergies WHERE patient_id = ?", (patient_id,)).fetchall()]
        meds = db.execute("SELECT * FROM refill_records WHERE patient_id = ?", (patient_id,)).fetchall()
        
        if "allergy" in message or "allergies" in message:
            if allergies:
                reply = f"Your medical vault contains recorded allergies to: **{', '.join(allergies)}**."
            else:
                reply = "No drug allergies are currently registered in your patient health record."
                
        elif "medicine" in message or "medication" in message or "taking" in message:
            if meds:
                med_lines = [f"- **{row['medicine_name']}** (Refill expected: {row['estimated_refill_date']})" for row in meds]
                reply = "Here are your active medications recorded in NALAM:\n" + "\n".join(med_lines)
            else:
                reply = "You do not have any active medications logged in your profile."
                
        elif "refill" in message or "remaining" in message:
            if meds:
                refill_lines = []
                for row in meds:
                    rem = row["remaining_quantity"]
                    daily = row["daily_quantity"]
                    days = rem // daily if daily > 0 else 0
                    refill_lines.append(f"- **{row['medicine_name']}**: {rem} units left ({days} days remaining, refill by: {row['estimated_refill_date']})")
                reply = "Estimated medication refill countdowns:\n" + "\n".join(refill_lines)
            else:
                reply = "No active medications found to compute refill dates."

        elif "prescription" in message or "rx" in message:
            # Query historical prescriptions from SQLite
            rx_rows = db.execute("""
                SELECT p.doctor_name, p.hospital_name, p.prescription_date, pi.medicine_name, pi.dosage_amount 
                FROM prescriptions p
                JOIN prescription_items pi ON p.id = pi.prescription_id
                WHERE p.patient_id = ?
                ORDER BY p.prescription_date DESC
            """, (patient_id,)).fetchall()
            if rx_rows:
                rx_lines = [f"- **{row['medicine_name']}** ({row['dosage_amount']}mg) issued by **{row['doctor_name']}** ({row['hospital_name']}) on {row['prescription_date']}" for row in rx_rows]
                reply = "Here is your historical prescriptions ledger:\n" + "\n".join(rx_lines)
            else:
                reply = "No historical prescriptions found in your NALAM medical ledger."

        elif "specialist" in message or "consult" in message or "hospital" in message or "doctor" in message:
            reply = "You can find local specialists and hospitals using our **Doctor & Hospital Finder** tab in your Patient Portal! It ranks recommended care based on your location, doctor availability, and specialty requirements using mathematical match scoring."
                
        elif "help" in message or "what can" in message:
            reply = "You can ask me questions about your recorded health records. Examples:\n" \
                    "- *'What medicines am I currently taking?'*\n" \
                    "- *'Do I have any allergies logged?'*\n" \
                    "- *'When is my next medication refill date?'*\n" \
                    "- *'Show my previous prescriptions.'*\n" \
                    "- *'Which specialist should I consult?'*"
        else:
            reply = "I am NALAM's local decision-support assistant. I can query your secure health vault for allergies, active prescriptions, and refill schedules. Type 'help' to see sample queries. Please note that I do not diagnose illnesses or write prescriptions."
            
    elif role == "Doctor":
        # Rule-based response for Doctors
        if "patient" in message or "records" in message:
            reply = "To access a patient's medical files, search for their registered email in your Doctor Portal, request clinical consent, and complete the OTP verification flow. Once approved, details will load automatically."
        else:
            reply = "Greetings, Doctor. I am the NALAM clinical assistant. I can guide you on the secure patient consent protocol or help search patient email records. For security, clinical access requires an active OTP authorization."
            
    db.close()
    return jsonify({"status": "success", "reply": reply})

# --- Model Metrics APIs ---

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    global risk_metrics, anomaly_metrics
    if risk_metrics is None or anomaly_metrics is None:
        load_ml_assets()
        
    return jsonify({
        "risk": risk_metrics or {"accuracy": "N/A"},
        "anomaly": anomaly_metrics or {"accuracy": "N/A"}
    })

# --- Emergency View API ---

@app.route("/api/emergency/<int:patient_id>", methods=["GET"])
def emergency_view(patient_id):
    db = get_db()
    p = db.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    
    if not p:
        db.close()
        return jsonify({"status": "error", "message": "Patient not found"}), 404
        
    # Emergency view gets limited critical records
    allergies = [row["allergy_name"] for row in db.execute("SELECT allergy_name FROM allergies WHERE patient_id = ?", (patient_id,)).fetchall()]
    meds = [row["medicine_name"] for row in db.execute("SELECT medicine_name FROM refill_records WHERE patient_id = ?", (patient_id,)).fetchall()]
    chronic = [row["disease_name"] for row in db.execute("SELECT disease_name FROM chronic_diseases WHERE patient_id = ?", (patient_id,)).fetchall()]
    
    db.close()
    return jsonify({
        "status": "success",
        "emergency_vault": {
            "patient_name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "blood_group": p["blood_group"],
            "critical_allergies": allergies,
            "major_medical_history": chronic,
            "current_medications": meds
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
