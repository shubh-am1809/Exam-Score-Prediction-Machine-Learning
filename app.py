# ===============================
# Exam Score Prediction App (CLI)
# ===============================

import pandas as pd
from src.utils import load_model

def main():
    model = load_model("models/exam_score_model.pkl")

    print("=== Exam Score Prediction ===")

    age = int(input("Enter Age: "))
    gender = input("Enter Gender (male/female): ").strip().lower()
    course = input("Enter Course (bca/btech/etc): ").strip().lower()
    study_hours = float(input("Enter Study Hours: "))
    class_attendance = float(input("Enter Attendance (%): "))
    internet_access = input("Internet Access (yes/no): ").strip().lower()
    sleep_hours = float(input("Enter Sleep Hours: "))
    sleep_quality = input("Sleep Quality (poor/average/good): ").strip().lower()
    study_method = input("Study Method (self/coaching/group): ").strip().lower()
    facility_rating = input("Facility Rating (low/medium/high): ").strip().lower()
    exam_difficulty = input("Exam Difficulty (easy/moderate/hard): ").strip().lower()

    new_student = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }])

    predicted_score = model.predict(new_student)
    print("\n Predicted Exam Score:", round(predicted_score[0], 2))

if __name__ == "__main__":
    main()
