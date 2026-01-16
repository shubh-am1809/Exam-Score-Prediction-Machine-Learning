# Exam Score Prediction (Machine Learning)

This project predicts **student exam scores** using multiple factors like:
study hours, attendance, sleep, course, study method, etc.

## Project Structure
```
Exam-Score-Prediction/
├── dataset/
│   └── Exam_Score_Prediction.csv
├── notebooks/
│   └── exam_score_prediction.ipynb
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
├── models/
│   └── exam_score_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Train the model
```bash
python src/train_model.py
```

## Predict exam score
```bash
python src/predict.py
```

## Run app
```bash
python app.py
```

