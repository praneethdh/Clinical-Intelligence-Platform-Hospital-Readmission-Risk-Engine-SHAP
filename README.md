# 🏥 Clinical Intelligence Platform: Hospital Readmission & Risk Stratification

An advanced, production-grade clinical decision support system designed to predict 30-day patient readmission risks. This platform bridges the gap between raw healthcare data and actionable medical insights using a transparent, multi-layered machine learning architecture.

## 🔗 Live Application Link

🚀 **Clinical AI Dashboard:** [Launch Full-Stack Application](https://www.google.com/search?q=YOUR_LINK_HERE)
(Fully containerized with Docker, featuring an integrated React/Tailwind frontend and FastAPI inference engine.)

## 🚀 Architecture

This system is architected for high-performance clinical environments:

* **Frontend:** Built with React, Tailwind CSS, and Vite. It provides a glassmorphic, responsive interface designed for clinical tablets and desktop workstations.
* **Backend:** A high-speed, asynchronous FastAPI service that enforces strict Pydantic data validation and serves model inferences.
* **AI Engine:** * **XGBoost Classifier:** The primary model, fine-tuned via Optuna for high-stakes binary classification.
* **Imbalance Management:** Calibrated using cost-sensitive learning (`scale_pos_weight`) to ensure the model captures the high-risk minority class effectively.
* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) to surface real-time risk drivers for every patient.



## 🌟 Key Features

* **Real-time Inference:** Sub-100ms risk scoring for point-of-care decision support.
* **Clinical Explainability:** Automatic generation of "Risk Factor" summaries, explaining exactly why a patient is flagged as high-risk.
* **Professional UI:** Modern, clean, and accessible design optimized for medical professionals to reduce cognitive load.
* **Robust Data Pipeline:** Automated feature engineering that maps raw diagnostic codes into clinical risk groupings.

## 🛠️ Getting Started

### Prerequisites

* Python 3.10+
* Node.js 18+
* Docker (Optional, for containerized deployment)

### 1. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload

```

The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev

```

The dashboard will be available at `http://localhost:5173`.

## 📂 Project Structure

```text
├── backend/              # FastAPI server, Pydantic schemas, Joblib artifacts
├── frontend/             # React + Tailwind + Vite implementation
├── notebooks/            # EDA, Training pipelines, and Optuna optimization
├── datasets/             # Cleaned clinical records (anonymized)
├── main.py               # API Entry point
└── requirements.txt      # Production environment dependencies

```

## 📈 System Objectives

* **Clinical Quality:** Reduce 30-day hospital readmission rates through early, data-driven intervention.
* **Transparency:** Eliminate "black-box" predictions by providing doctors with SHAP-based diagnostic attributions.
* **Operational Efficiency:** Automate clinical risk assessment to save staff time and focus resources on critical patients.
