# AI-Based Diabetes Diagnosis & Retinopathy Detection System

> An end-to-end AI diagnostic system integrating Deep Learning and Machine Learning models to predict diabetes and detect diabetic retinopathy through a unified web interface.

---

## Overview

This system addresses three clinical diagnostic tasks using a combination of CNN-based deep learning and classical machine learning models, all deployed through a Flask web application for real-time inference.

| Diagnostic Task | Input Type | Model | Accuracy |
|---|---|---|---|
| Diabetes Prediction | Structured medical data | XGBoost | 97% |
| Gestational Diabetes Prediction | Structured obstetric data | XGBoost | 98% |
| Diabetic Retinopathy Detection | Retinal fundus images | CNN + Random Forest | ~70% |

---

## Model Architecture

### Diabetic Retinopathy — Two-Stage Hybrid Pipeline

```
Retinal Image
     │
     ▼
[ CNN — Feature Extraction ]
     │
     ▼
[ Random Forest Classifier ]
     │
     ├──▶ Stage 1: Binary Detection    (Positive / Negative)
     │
     └──▶ Stage 2: Severity Grading   (Mild / Moderate / Severe / Proliferative)
```

### Diabetes & Gestational Diabetes — XGBoost Classifiers

```
Clinical Features (Glucose, BMI, Insulin, Age ...)
     │
     ▼
[ XGBoost Classifier ]
     │
     └──▶ Risk Prediction Output
```

---

## Technology Stack

```
Deep Learning      TensorFlow · Keras
Machine Learning   Scikit-learn · XGBoost
Image Processing   OpenCV
Web Framework      Flask
Data Processing    NumPy · Pandas · Matplotlib
Language           Python 3.9+
```

---

## Project Structure

```
Diabetes-Diagnosis-System/
│
├── app.py                  # Flask application — integrates all models
├── requirements.txt        # Project dependencies
│
├── models/                 # Saved model files (.h5, .pkl)
├── notebooks/              # Model development and training notebooks
├── dataset/                # Image and tabular training datasets
│
├── static/                 # CSS, JavaScript, image assets
└── templates/              # HTML templates for the web interface
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/Diabetes-Diagnosis-System.git
cd Diabetes-Diagnosis-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py

# 4. Open in browser
http://127.0.0.1:5000
```

---

## Future Work

- Integration with hospital EMR systems for clinical deployment
- Patient history dashboard for longitudinal monitoring
- Cloud API deployment and mobile application support
- Retinal feature extraction enhancement using VGG16 and ResNet
- AI-assisted conversational interface for guided diagnosis

---

## Author

**Norhan Kamal Hosny** — AI / ML Engineer

Machine Learning · Deep Learning · Intelligent Diagnostic Systems
