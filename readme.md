# Symptom Checker

An educational machine learning web application built with **FastAPI** and **scikit-learn** that suggests likely conditions from selected symptoms, along with disease descriptions, symptom severity weights, and recommended precautions.

> **Disclaimer:** Educational project using a public dataset. This is not medical advice and cannot diagnose any condition. Please consult a doctor.

## Features

- **Symptom-Based Disease Prediction:** Evaluates user-selected symptoms against a trained machine learning model (`disease_model.pkl`).
- **Model Evaluation & Training Pipeline:** Cleans raw symptom and disease labels and evaluates `RandomForestClassifier`, `BernoulliNB`, and `DecisionTreeClassifier` models using stratified k-fold cross-validation (`StratifiedKFold`).
- **Condition Details & Precautions:** Enriches predictions using structured datasets covering disease descriptions, symptom severity weights, and precaution checklists.
- **FastAPI Backend & Web UI:** Serves both the frontend interface (`index.html`) and REST API endpoints with interactive Swagger documentation.

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend API** | Python, FastAPI, Uvicorn, Pydantic |
| **Machine Learning & Data** | scikit-learn, pandas, NumPy, joblib |
| **Frontend** | HTML (`index.html`) |

## Project Structure

- `data/` — Public CSV datasets used for training and metadata lookup:
  - `dataset.csv` — Disease-to-symptom training records
  - `symptom_Description.csv` — Overview descriptions for each condition
  - `symptom_precaution.csv` — Recommended precautionary steps per condition
  - `Symptom-severity.csv` — Severity weights for individual symptoms
- `models/` — Serialized model artifacts:
  - `disease_model.pkl` — Trained model and metadata bundle loaded at startup
- `src/` — Application and training source code:
  - `main.py` — FastAPI application, lifespan model loader, and API routes
  - `train_model.py` — Data preprocessing, model cross-validation, and artifact export script
- `index.html` — Web interface for selecting symptoms and viewing results
- `.gitignore` — Git ignore rules for Python cache and local environment files

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/zahid-06/Symptom-Based-Disease-Prediction-System
cd Symptom-Based-Disease-Prediction-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Model

The repository already includes a pre-trained model bundle in `models/disease_model.pkl`. To re-train or update the model:

```bash
py src/train_model.py
```

### 4. Run the Application

Start the FastAPI server from the project root folder:

```bash
py -m uvicorn src.main:app --reload
```

Once the server is running:
- **Web Interface:** Open `http://127.0.0.1:8000` in your browser
- **Interactive API Docs (Swagger UI):** Open `http://127.0.0.1:8000/docs`
