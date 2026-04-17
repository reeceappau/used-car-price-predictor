# Bento Motors — Used Car Price Predictor

A machine learning pipeline for predicting used car prices, developed as part of the Applied AI module at University Academy 92.

---

## Project Overview

This project builds a supervised machine learning pipeline trained on over 400,000 UK vehicle listings. A Random Forest model is used to predict listing prices based on features such as vehicle age, mileage, make, body type, and fuel type. The model achieves a test R² of 0.8866.

The pipeline includes:
- Exploratory data analysis and feature engineering
- Data preprocessing and model training
- Hyperparameter tuning using RandomizedSearchCV
- Model interpretation using SHAP
- An interactive Streamlit web application for price prediction
- Cloud deployment on Render using Docker and Terraform

---

## Repository Structure

```
├── bento_motors_ml.ipynb     # Full ML pipeline notebook
├── app.py                    # Streamlit web application
├── Dockerfile                # Docker environment configuration
├── requirements.txt          # Python dependencies
├── main.tf                   # Terraform infrastructure configuration
├── adverts.csv               # Dataset (UK vehicle listings)
└── README.md                 # This file
```

---

## Running the Notebook

### Prerequisites

- Python 3.11
- Jupyter Notebook or VS Code with Jupyter extension

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib scipy
```

### Run the notebook

Open `bento_motors_ml.ipynb` and run all cells from top to bottom. The notebook will:

1. Load and explore the dataset
2. Preprocess the data
3. Train and tune the models
4. Evaluate performance
5. Generate SHAP explanations
6. Save `model.pkl`, `scaler.pkl`, and `feature_names.pkl`

---

## Running the Streamlit App Locally

### Prerequisites

Ensure the notebook has been run first so that `model.pkl`, `scaler.pkl`, and `feature_names.pkl` exist in the project folder.

### Install Streamlit

```bash
pip install streamlit
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Enter vehicle details and click **Predict price** to receive an estimated price and a SHAP explanation.

---

## Cloud Deployment with Docker and Terraform

The cloud deployment hosts the Streamlit app on Render using Docker for environment reproducibility and Terraform for Infrastructure as Code.

### Prerequisites

- [Docker](https://www.docker.com/) installed
- [Terraform](https://www.terraform.io/) installed (`brew install terraform` on Mac)
- A [Render](https://render.com/) account with an API key

### Step 1 — Build and test Docker image locally (optional)

```bash
docker build -t bento-motors .
docker run -p 8501:8501 bento-motors
```

Visit `http://localhost:8501` to confirm the app runs correctly inside the container.

### Step 2 — Push code to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### Step 3 — Configure Terraform credentials

Create a `terraform.tfvars` file in the project root (this file is gitignored and should never be committed):

```hcl
render_api_key  = "your-render-api-key"
render_owner_id = "your-render-owner-id"
```

To find your Render API key and owner ID:
- API key: Render dashboard → Account Settings → API Keys → Create API Key
- Owner ID: Render dashboard → Account Settings → shown in the URL or user settings

### Step 4 — Provision the Render service with Terraform

```bash
terraform init
terraform plan
terraform apply
```

Type `yes` when prompted. Terraform will provision the Render web service automatically. The live URL will be available in the Render dashboard once the build completes.

> **Note:** The free tier on Render provides 512MB RAM. The deployed app retrains on a 50,000-row sample on startup rather than loading the full model. The service may take 2-3 minutes to wake up after a period of inactivity.

