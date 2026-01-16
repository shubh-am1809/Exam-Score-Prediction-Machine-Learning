# ===============================
# Predict Exam Score
# ===============================

import pandas as pd
from utils import load_model

# -------------------------------
# 1. Load Saved Model
# -------------------------------
model_path = "models/exam_score_model.pkl"
model = load_model(model_path)

# -------------------------------
# 2. New Student Data (Example)
# -------------------------------
new_student = pd.DataFrame([{
    "age": 20,
    "gender": "male",
    "course": "bca",
    "study_hours": 6.5,
    "class_attendance": 85,
    "internet_access": "yes",
    "sleep_hours": 7,
    "sleep_quality": "good",
    "study_method": "coaching",
    "facility_rating": "high",
    "exam_difficulty": "moderate"
}])

# -------------------------------
# 3. Predict
# -------------------------------
predicted_score = model.predict(new_student)
print("✅ Predicted Exam Score:", round(predicted_score[0], 2))
