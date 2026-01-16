# ===============================
# Train Exam Score Prediction Model
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import save_model

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("dataset/Exam_Score_Prediction.csv")

# -------------------------------
# 2. Define Features & Target
# -------------------------------
X = df.drop(["exam_score", "student_id"], axis=1)
y = df["exam_score"]

# -------------------------------
# 3. Separate Column Types
# -------------------------------
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

# -------------------------------
# 4. Preprocessing Pipeline
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# -------------------------------
# 5. Model
# -------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# -------------------------------
# 6. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. Train Model
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# 8. Evaluate Model
# -------------------------------
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Training Completed!")
print("-----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# -------------------------------
# 9. Save Model
# -------------------------------
model_path = "models/exam_score_model.pkl"
save_model(pipeline, model_path)

print(f"\n✅ Model Saved Successfully at: {model_path}")
