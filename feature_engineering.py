import numpy as np

# Mock medication interaction database for prototype validation (MathX project requirement)
DRUG_INTERACTIONS = {
    frozenset(["Warfarin", "Aspirin"]): 3.0,
    frozenset(["Sildenafil", "Nitroglycerin"]): 5.0,
    frozenset(["Simvastatin", "Amlodipine"]): 1.5,
    frozenset(["Ibuprofen", "Warfarin"]): 3.0,
    frozenset(["Lisinopril", "Spironolactone"]): 2.0,
    frozenset(["Aspirin", "Ibuprofen"]): 1.0
}

def calculate_interaction_score(medications):
    """
    Calculates interaction risk score as: sum(interaction_indicator * severity_score)
    medications: list of medication names (strings)
    """
    if not medications or len(medications) < 2:
        return 0.0
        
    score = 0.0
    n = len(medications)
    for i in range(n):
        for j in range(i + 1, n):
            med1 = medications[i].strip().title()
            med2 = medications[j].strip().title()
            pair = frozenset([med1, med2])
            if pair in DRUG_INTERACTIONS:
                score += DRUG_INTERACTIONS[pair]
    return score

def compute_zscore(value, mean, std):
    """
    Z = (x - mean) / standard_deviation
    """
    if std == 0:
        return 0.0
    return (value - mean) / std

def calculate_adherence(doses_taken, doses_expected):
    """
    Adherence (%) = Doses Taken / Doses Expected * 100
    """
    if doses_expected == 0:
        return 100.0
    return min((doses_taken / doses_expected) * 100.0, 100.0)
