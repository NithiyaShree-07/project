import os
import sqlite3
from werkzeug.security import generate_password_hash

def init_database():
    db_dir = "database"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    db_path = os.path.join(db_dir, "database.db")
    print(f"Initializing SQLite database at '{db_path}'...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('Patient', 'Doctor'))
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        blood_group TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        specialty TEXT NOT NULL,
        hospital TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS allergies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        allergy_name TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chronic_diseases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        disease_name TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prescriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        doctor_name TEXT NOT NULL,
        hospital_name TEXT NOT NULL,
        prescription_date TEXT NOT NULL,
        uploaded_file TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prescription_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prescription_id INTEGER NOT NULL,
        medicine_name TEXT NOT NULL,
        dosage_amount INTEGER NOT NULL,
        dosage_frequency INTEGER NOT NULL,
        treatment_duration INTEGER NOT NULL,
        refill_delay_days INTEGER NOT NULL,
        prescription_frequency INTEGER NOT NULL,
        interaction_count INTEGER NOT NULL DEFAULT 0,
        adherence INTEGER NOT NULL DEFAULT 100,
        FOREIGN KEY (prescription_id) REFERENCES prescriptions(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS consent_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('Pending', 'Approved', 'Rejected', 'Revoked')),
        otp TEXT,
        otp_verified INTEGER DEFAULT 0,
        share_medical_history INTEGER DEFAULT 0,
        share_medications INTEGER DEFAULT 0,
        share_allergies INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        expiry_at TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
        FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER NOT NULL,
        patient_id INTEGER NOT NULL,
        access_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        details TEXT NOT NULL,
        FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE CASCADE,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS refill_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        medicine_name TEXT NOT NULL,
        remaining_quantity INTEGER NOT NULL,
        daily_quantity INTEGER NOT NULL,
        estimated_refill_date TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        prescription_id INTEGER,
        risk_level TEXT NOT NULL,
        confidence REAL NOT NULL,
        anomaly_status TEXT NOT NULL,
        anomaly_score REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
        FOREIGN KEY (prescription_id) REFERENCES prescriptions(id) ON DELETE SET NULL
    );
    """)
    
    # 2. Seed Default Accounts
    print("Seeding default Patient and Doctor accounts...")
    
    # Check if seeds already exist
    cursor.execute("SELECT id FROM users WHERE email = 'patient@example.com'")
    if not cursor.fetchone():
        # Insert Patient User
        cursor.execute("""
        INSERT INTO users (username, email, password_hash, role)
        VALUES ('patient_user', 'patient@example.com', ?, 'Patient')
        """, (generate_password_hash("patient123"),))
        patient_user_id = cursor.lastrowid
        
        # Insert Patient Detail Profile
        cursor.execute("""
        INSERT INTO patients (user_id, name, age, gender, blood_group)
        VALUES (?, 'John Doe', 67, 'Male', 'O+')
        """, (patient_user_id,))
        patient_id = cursor.lastrowid
        
        # Insert Patient Allergies
        cursor.execute("INSERT INTO allergies (patient_id, allergy_name) VALUES (?, 'Penicillin')", (patient_id,))
        cursor.execute("INSERT INTO allergies (patient_id, allergy_name) VALUES (?, 'Sulfa Drugs')", (patient_id,))
        
        # Insert Patient Chronic Diseases
        cursor.execute("INSERT INTO chronic_diseases (patient_id, disease_name) VALUES (?, 'Hypertension')", (patient_id,))
        cursor.execute("INSERT INTO chronic_diseases (patient_id, disease_name) VALUES (?, 'Type 2 Diabetes')", (patient_id,))
        
        # Seed historical prescriptions and items (Normal pattern)
        cursor.execute("""
        INSERT INTO prescriptions (patient_id, doctor_name, hospital_name, prescription_date, uploaded_file)
        VALUES (?, 'Dr. Sarah Smith', 'Metro General Hospital', '2026-07-01', 'sample_rx1.pdf')
        """, (patient_id,))
        rx_id = cursor.lastrowid
        
        cursor.execute("""
        INSERT INTO prescription_items (prescription_id, medicine_name, dosage_amount, dosage_frequency, treatment_duration, refill_delay_days, prescription_frequency, interaction_count, adherence)
        VALUES (?, 'Metformin', 500, 2, 30, 1, 12, 0, 95)
        """, (rx_id,))
        
        cursor.execute("""
        INSERT INTO prescription_items (prescription_id, medicine_name, dosage_amount, dosage_frequency, treatment_duration, refill_delay_days, prescription_frequency, interaction_count, adherence)
        VALUES (?, 'Lisinopril', 10, 1, 30, 2, 12, 0, 90)
        """, (rx_id,))
        
        # Seed Refill Records
        cursor.execute("""
        INSERT INTO refill_records (patient_id, medicine_name, remaining_quantity, daily_quantity, estimated_refill_date)
        VALUES (?, 'Metformin', 20, 2, '2026-08-10')
        """, (patient_id,))
        cursor.execute("""
        INSERT INTO refill_records (patient_id, medicine_name, remaining_quantity, daily_quantity, estimated_refill_date)
        VALUES (?, 'Lisinopril', 5, 1, '2026-08-04')
        """, (patient_id,))
        
        print("Patient 'patient@example.com' seeded successfully.")
        
    cursor.execute("SELECT id FROM users WHERE email = 'doctor@example.com'")
    if not cursor.fetchone():
        # Insert Doctor User
        cursor.execute("""
        INSERT INTO users (username, email, password_hash, role)
        VALUES ('doctor_user', 'doctor@example.com', ?, 'Doctor')
        """, (generate_password_hash("doctor123"),))
        doctor_user_id = cursor.lastrowid
        
        # Insert Doctor Profile
        cursor.execute("""
        INSERT INTO doctors (user_id, name, specialty, hospital)
        VALUES (?, 'Dr. Sarah Smith', 'Cardiology', 'Metro General Hospital')
        """, (doctor_user_id,))
        
        print("Doctor 'doctor@example.com' seeded successfully.")
        
    conn.commit()
    conn.close()
    print("Database initialization complete.")

if __name__ == "__main__":
    init_database()
