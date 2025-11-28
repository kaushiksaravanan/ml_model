import csv
import random
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
NUM_ROWS = 600

cities_with_pins = {
    "Hyderabad": [500001, 500002, 500003, 500004, 500005],
    "Bengaluru": [560001, 560002, 560003, 560004, 560005],
    "Chennai": [600001, 600002, 600003, 600004, 600005],
    "Mumbai": [400001, 400002, 400003, 400004, 400005],
    "Pune": [411001, 411002, 411003, 411004, 411005]
}

specialties = [
    "Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Gynecology",
    "Dermatology", "Oncology", "ENT", "General Medicine",
    "Nephrology", "Urology", "Gastroenterology", "Pulmonology",
    "Emergency Medicine", "ICU Care"
]

hospital_names = [
    "Apollo Hospitals", "Fortis Healthcare", "Max Super Speciality Hospital",
    "Columbia Asia Hospital", "Aster Prime Hospital", "Rainbow Children’s Hospital",
    "Yashoda Hospital", "Continental Hospitals", "Narayana Health",
    "Care Hospitals", "KIMS Hospital", "Manipal Hospital", "Sunshine Hospitals",
    "Star Hospitals", "Global Hospitals", "Medicover Hospital",
    "Vasan Eye Care", "Omega Hospitals", "Vinn Hospital",
    "Ankura Hospital", "Prerana Hospital", "Trinity Healthcare",
    "Pulse Multi‑Speciality Hospital", "City Care Hospital",
    "Lifeline Hospital", "Heritage Medical Center", "Sanjeevani Hospital",
    "Artemis Medicare", "Nexus Hospital", "Prime Care Hospital",
    "Horizon Super Speciality", "Unity Hospital", "Revive Medical Center",
    "Grace Care Hospital", "Elite Multi‑Speciality Hospital",
    "Harmony Health Center", "MetroCare Hospital", "Wellness Hospital",
    "CureOne Medical Center", "Vibrant Life Hospital"
]

def rand_phone():
    return f"+91{random.randint(6000000000, 9999999999)}"

# ---------------------------
# GENERATE CSV
# ---------------------------

filename = "hospital_details.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "hospital_id", "hospital_name", "city", "pincode",
        "specialty",
        "total_beds", "icu_beds", "ventilators",
        "doctors_available", "nurses_available",
        "oxygen_cylinders", "ppe_kits",
        "emergency_available",
        "rating",
        "contact_number",
        "last_updated"
    ])

    for i in range(1, NUM_ROWS + 1):
        city = random.choice(list(cities_with_pins.keys()))
        pincode = random.choice(cities_with_pins[city])

        total_beds = random.randint(30, 400)
        icu_beds = random.randint(5, max(5, total_beds // 5))
        ventilators = random.randint(2, icu_beds)

        writer.writerow([
            i,
            random.choice(hospital_names),
            city,
            pincode,
            random.choice(specialties),
            total_beds,
            icu_beds,
            ventilators,
            random.randint(5, 50),      # doctors
            random.randint(10, 120),    # nurses
            random.randint(10, 200),    # oxygen cylinders
            random.randint(20, 400),    # PPE kits
            random.choice(["Yes", "No"]),
            round(random.uniform(3.0, 5.0), 1),
            rand_phone(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

print(f"\nCSV file '{filename}' generated successfully with {NUM_ROWS} rows!\n")
