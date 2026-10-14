# 🛡️ Fraud Detection MLOps

An end-to-end machine learning project for detecting potentially fraudulent financial transactions using a Random Forest classifier, with a FastAPI backend and a React-based dashboard.

The project demonstrates how machine learning can be integrated into an application for transaction scoring, prediction serving, and fraud analytics.

## ✨ Features

- **Fraud Prediction API** — Predict whether a transaction is potentially fraudulent or legitimate.
- **Random Forest Model** — Generate fraud probabilities from transaction features.
- **Configurable Decision Threshold** — Classify transactions based on a probability threshold.
- **FastAPI Backend** — Serve predictions through REST API endpoints.
- **Interactive API Documentation** — Test endpoints using Swagger UI.
- **Transaction Analytics** — Retrieve overview analytics through the analytics endpoint.
- **React + Vite Frontend** — Web interface for interacting with the fraud detection system.
- **Model Health Check** — Check API status and whether the model is loaded.
- **MLOps-Oriented Structure** — Organize model artifacts, datasets, configuration, and application code for further development.

## 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Language | Python, JavaScript |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | Pandas, NumPy |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Visualization | Recharts |
| Model Serialization | Joblib |
| MLOps / Experiment Tracking | MLflow |
| Containerization | Docker |
| Testing | Pytest |

*Some listed tools are part of the project structure or dependencies; their full integration should be verified before claiming every workflow is operational.*

## 🏗️ Project Structure

```text
Fraud-Detection-MLOps/
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes/
│       ├── ml/
│       └── main.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── candidates/
├── ml/
├── mlops/
├── configs/
├── notebooks/
├── scripts/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
└── README.md
```

## 🔄 How It Works

1. A user submits transaction details through the frontend or API.
2. The backend validates the incoming transaction data.
3. The model service loads the trained Random Forest artifact.
4. The model calculates a fraud probability.
5. The probability is compared with the configured decision threshold.
6. The API returns the prediction, fraud probability, threshold, and model name.
7. The analytics endpoint provides an overview of the transaction dataset.

## 📊 Dataset

This project uses the **PaySim synthetic financial transaction dataset**, designed to simulate mobile-money transactions and fraudulent activity.

Typical fields include:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`

The `isFraud` column is the target label used during model training. It should not be supplied as a prediction feature.

**Dataset note:** The full PaySim dataset is large. Keep raw datasets and trained model artifacts out of Git unless you intentionally want to distribute them and have checked repository limits and licensing.

## ⚙️ Prerequisites

Install the following before running the project:

- Python 3.11 or another Python version compatible with the project's dependencies
- Node.js and npm
- Git

Docker is optional if you intend to run the containerized setup.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Fraud-Detection-MLOps.git
cd Fraud-Detection-MLOps
```

Replace `YOUR_USERNAME` with your GitHub username and adjust the repository name if needed.

### 2. Set up the backend

From the project root, create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
python -m pip install -r backend/requirements.txt
```

### 3. Start the FastAPI server

Run this command from the project root:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open the API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

The trained model artifact must be present at the path expected by the model service before predictions can work.

### 4. Set up the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local frontend URL printed by Vite, usually http://localhost:5173.

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API home |
| `GET` | `/api/health` | Check service and model status |
| `POST` | `/api/predict` | Predict fraud probability and classification |
| `GET` | `/api/analytics/overview` | Retrieve transaction analytics |

### Example prediction request

**POST** `/api/predict`

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 5000,
  "oldbalanceOrg": 5000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 1000,
  "newbalanceDest": 6000
}
```

### Example response

```json
{
  "prediction": "LEGITIMATE",
  "is_fraud": false,
  "fraud_probability": 0.77036,
  "threshold": 0.95,
  "model_name": "Random Forest",
  "message": "Transaction classified as legitimate."
}
```

*This is an example response from a test request. Predictions depend on the input, trained model, and configured threshold.*

## 🩺 Health Check

Example response from `/api/health`:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Random Forest",
  "threshold": 0.95
}
```

## 🧪 Testing

If the project's tests are configured, run them from the root directory:

```powershell
python -m pytest
```

You can also use Swagger UI to test the prediction and health endpoints interactively.

## 🔐 Best Practices

- Do not commit `.env` files, credentials, or API keys.
- Avoid committing large datasets, virtual environments, and `node_modules`.
- Store model artifacts separately when they exceed Git hosting limits.
- Validate and sanitize incoming transaction data.
- Evaluate precision, recall, F1-score, and precision-recall AUC because fraud datasets are highly imbalanced.
- Choose the classification threshold based on business requirements and the cost of false positives versus false negatives.
- Treat model predictions as decision-support signals, not definitive proof of fraud.

## 🛣️ Future Improvements

- Add automated model retraining and validation.
- Integrate model and data drift monitoring.
- Track experiments and model versions through MLflow.
- Add automated testing and CI/CD with GitHub Actions.
- Improve dashboard analytics and model performance reporting.
- Add authentication, structured logging, and production deployment monitoring.

## 👩‍💻 Author

**Vaishnavi Kirkan**

GitHub: [VaishnaviKirkan01](https://github.com/VaishnaviKirkan01)

---

⭐ If you find this project useful, consider starring the repository.

*Built as a practical machine learning and MLOps project to explore fraud detection, model serving, and application development.*
