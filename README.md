# AI-Based Diabetes Diagnosis and Retinopathy Detection System

## Overview

This project is an AI-powered diagnostic system designed to predict and detect diabetes and its associated complications using a combination of Deep Learning and Machine Learning models. The system addresses three clinical diagnostic tasks:

- **Diabetes Prediction** — Assesses diabetes risk based on patient medical data including glucose levels, BMI, insulin, and related clinical features.
- **Gestational Diabetes Prediction** — Evaluates gestational diabetes risk in pregnant patients using specialized medical indicators.
- **Diabetic Retinopathy Detection** — Analyzes retinal fundus images to detect the presence of diabetic retinopathy and classify disease severity.

All models are integrated into a unified Flask web application, enabling users to input medical data or upload retinal images and receive immediate diagnostic predictions.

---

## Model Architecture

### Diabetic Retinopathy Detection
A two-stage hybrid pipeline combining Convolutional Neural Networks (CNN) for deep feature extraction with a Random Forest classifier for final classification.

- **Stage 1 — Binary Classification:** Detects the presence or absence of diabetic retinopathy.
- **Stage 2 — Severity Classification:** Categorizes confirmed cases as Mild, Moderate, Severe, or Proliferative Diabetic Retinopathy.
- Overall accuracy: approximately 70%.

### Diabetes Prediction
An XGBoost classifier trained on structured clinical data, achieving 97% accuracy in predicting diabetes risk.

### Gestational Diabetes Prediction
A dedicated XGBoost model trained on obstetric and metabolic features specific to pregnant patients, achieving 98% accuracy.

---

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn, XGBoost |
| Web Framework | Flask |
| Image Processing | OpenCV |
| Data Processing | NumPy, Pandas, Matplotlib |

---

## Key Features

- Multi-model diagnostic system supporting both image-based and numerical data inputs
- Hybrid Deep Learning and Machine Learning architecture
- Flask-based interactive web interface with real-time prediction output
- End-to-end pipeline covering preprocessing, inference, and result delivery

---

## Project Structure

```
Diabetes-Diagnosis-System/
├── app.py                  # Flask application integrating all models
├── models/                 # Saved ML and DL model files (.h5, .pkl)
├── static/                 # CSS, JavaScript, and uploaded image assets
├── templates/              # HTML templates for the web interface
├── dataset/                # Image and tabular datasets used for training
├── notebooks/              # Jupyter notebooks for model development
└── requirements.txt        # Project dependencies
```

---

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/Diabetes-Diagnosis-System.git
cd Diabetes-Diagnosis-System
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
python app.py
```

**4. Access the interface**
```
http://127.0.0.1:5000
```

---

## Future Work

- Integration with hospital Electronic Medical Record (EMR) systems for clinical deployment
- Development of a patient history dashboard for longitudinal tracking
- Cloud-based API deployment and mobile application support
- Enhancement of retinal feature extraction using pre-trained architectures such as VGG16 and ResNet
- Addition of an AI-assisted conversational interface for guided diagnosis

---

## Author

**Norhan Kamal Hosny**
AI / ML Engineer | Machine Learning and Intelligent Diagnostic Systems
