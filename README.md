# SME Risk Intelligence Platform

> An AI-powered financial risk assessment system for Small & Medium Enterprises (SMEs), built with Python, Streamlit, and machine learning.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red) ![ML](https://img.shields.io/badge/Model-Gradient%20Boosting-green) ![ROC AUC](https://img.shields.io/badge/ROC%20AUC-98.04%25-brightgreen) ![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2) ![Database](https://img.shields.io/badge/Database-Supabase-3ECF8E)

---

## 🔗 Live Demo

**Try the app right now — no setup required:**

### 👉 [https://sme-risk-prediction.streamlit.app/](https://sme-risk-prediction.streamlit.app/)

This is the fully deployed, functional version of the platform. If you're evaluating this project, this link takes you straight to the live system — dashboard, single & batch prediction, SHAP explainability, and the analytics dashboard are all working.

A ready-to-use sample file for testing the **Batch Prediction** feature is included in the repo at `Data/sample_batch_demo.csv` — just download it and upload it on the Prediction page.

---

## 📊 Project Infographic

As requested for the final graduation submission, a comprehensive one-page infographic summarizing the platform's architecture, ML pipeline, and core features has been prepared.

### 👉 [Download / View Project Infographic (PDF)](https://raw.github.com/AmrMahmoudFekry/DEPI-Project/blob/main/assets/infographic.pdf)

*(You can click the link above or use the interactive button below to view the PDF directly inside GitHub or download it)*

<p align="center">
  <a href="https://raw.githubusercontent.com/AmrMahmoudFekry/DEPI-Project/main/assets/infographic.pdf" target="_blank">
    <img src="https://img.shields.io/badge/View_&_Download_Infographic-PDF_Format-FF6B6B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Download PDF Infographic" height="50">
  </a>
</p>

---

## What This Project Does

The SME Risk Intelligence Platform helps banks and credit analysts make better lending decisions. Given a set of financial indicators for a business, the system:

- Predicts the **risk level** (Low / Medium / High) with a probability score, using a **Gradient Boosting** model selected out of 7 candidates via 10-fold cross-validation, with hyperparameters tuned through `RandomizedSearchCV`
- Explains **why** the model made that prediction (SHAP-based explainability, single business & portfolio-level)
- Generates **strategic recommendations** using rule-based logic and Google Gemini AI
- Produces a **downloadable PDF report** with a full bilingual risk assessment
- Supports **batch scoring** — upload a CSV of multiple businesses and score all at once
- Provides an **analytics dashboard** with model performance, feature importance, and portfolio insights
- Fully **bilingual** — English and Arabic (with RTL layout support)
- Tracks every training run and model version with **MLflow**
- Persists prediction history in a **Supabase (PostgreSQL) database** instead of a local file

---

## Project Structure

```
SME-Risk-Intelligence/
├── app.py
├── modeling_pipeline.py
├── requirements.txt
├── requirements_local.txt
├── .env.example
├── .gitignore
│
├── Data/
│   ├── SMEs_Data.csv
│   └── sample_batch_demo.csv
│
├── Notebooks/
│   ├── cleaning.ipynb
│   ├── Hyperparameter_Tuning.ipynb
│   └── Visualization.ipynb
│
├── Documentation/
│
├── utils/
│   ├── model_loader.py
│   ├── helper.py
│   ├── history_manager.py
│   └── db.py
│
├── components/
│   ├── charts.py
│   ├── analytics.py
│   ├── recommendation_engine.py
│   ├── ai_recommendation_engine.py
│   ├── ai_insights.py
│   ├── report_generator.py
│   ├── risk_badge.py
│   ├── ui_cards.py
│   └── dashboard_stats.py
│
├── assets/
│   ├── styles.css
│   ├── dark mode_Logo.png
│   ├── light mode_Logo.png
│   └── animation.json
│
├── pipeline.pkl
├── model_comparison.csv
├── model_comparison_results.json
├── mlflow.db
└── mlruns/
```

**Notes on a few entries:**

- `Data/sample_batch_demo.csv` — a ready-made file to try Batch Prediction instantly, without preparing your own CSV.
- `Notebooks/` — `cleaning.ipynb` (data cleaning & winsorization), `Hyperparameter_Tuning.ipynb` (RandomizedSearchCV tuning for all 7 models), `Visualization.ipynb` (exploratory data visualization).
- `utils/db.py` — a shared, cached Supabase client used by `history_manager.py` and the rest of the app.
- `pipeline.pkl`, `model_comparison.csv`, `model_comparison_results.json`, `mlflow.db`, `mlruns/` — all generated automatically by running `modeling_pipeline.py`; they aren't written by hand.
- `requirements.txt` (Streamlit Cloud) vs `requirements_local.txt` (full local dev, adds MLflow and other training-only dependencies) are kept separate so the cloud deployment stays lightweight.

---

## Quick Start (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sme-risk-intelligence.git
cd sme-risk-intelligence
```

### 2. Install Dependencies

For the full local experience (training, MLflow, everything):

```bash
pip install -r requirements_local.txt
```

For a lightweight setup matching what Streamlit Cloud uses:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example env file and fill in your own credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY="your_gemini_api_key_here"
SUPABASE_URL="your_supabase_project_url"
SUPABASE_API="your_supabase_service_or_anon_key"
```

> **Never commit your real `.env` file.** It's already listed in `.gitignore`.

### 4. Set Up Supabase (Database)

The app stores prediction history in Supabase instead of a local CSV.

1. Create a free project at [supabase.com](https://supabase.com).
2. In the Supabase SQL editor, create the `prediction_history` table:
   ```sql
   create table prediction_history (
       id bigint generated always as identity primary key,
       timestamp text,
       risk_score float8,
       confidence float8,
       risk_label text
   );
   ```
3. (Optional) If you want the Analytics page to read the dataset from Supabase instead of the local CSV fallback, also create a `smes_data` table with the same columns as `Data/SMEs_Data.csv` and import the CSV into it.
4. Copy your **Project URL** and **API key** (anon or service key) from Supabase → Settings → API.
5. Add them to `.streamlit/secrets.toml` for local Streamlit runs:
   ```toml
   SUPABASE_URL = "your_supabase_project_url"
   SUPABASE_KEY = "your_supabase_api_key"
   ```
   (The app reads these via `st.secrets` in `utils/db.py`.)

> If Supabase is unreachable or not configured, the Analytics page and dashboard summary fall back to the local `Data/SMEs_Data.csv` and an empty history so the app doesn't crash — but prediction history won't persist without Supabase.

### 5. Train the Model

> **Required before first run.** This generates `pipeline.pkl`, `model_comparison.csv`, `model_comparison_results.json`, `mlflow.db`, and `mlruns/`.

```bash
python modeling_pipeline.py
```

The script will:
- Load `Data/SMEs_Data.csv`
- Train and compare **7 ML models** (Gradient Boosting, XGBoost, Hist Gradient Boosting, Random Forest, Extra Trees, AdaBoost, Logistic Regression) using their already-tuned hyperparameters (from `Notebooks/Hyperparameter_Tuning.ipynb`)
- Track every run in **MLflow** (SQLite-backed tracking store, required for Model Registry support)
- Select the best model by 10-fold cross-validated ROC-AUC
- Calibrate it (`CalibratedClassifierCV`, isotonic, 5-fold)
- Save the calibrated pipeline to `pipeline.pkl`
- Register the model in the MLflow Model Registry and promote it to `Production`

To inspect all tracked runs afterwards:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000`.

### 6. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 7. Try Batch Prediction Instantly

Upload `Data/sample_batch_demo.csv` on the **Prediction → Batch Prediction** section of the app to see the full batch scoring flow, SHAP portfolio-level analysis, and downloadable batch report — no need to prepare your own file first.

---

## Enabling Gemini AI Recommendations (Optional)

The AI advisory feature uses Google Gemini. Without a key, the rest of the app works normally — only the AI advisory section will show a fallback message.

**Local setup:**

1. Make sure `GEMINI_API_KEY` is set in your `.env` file (see step 3 above).
2. Get a free API key at: https://aistudio.google.com/app/apikey

---

## Deployment on Streamlit Cloud

This project is already deployed and live at:

### 👉 [https://sme-risk-prediction.streamlit.app/](https://sme-risk-prediction.streamlit.app/)

To deploy your own copy:

1. Push your project to a GitHub repository (without `.env`, or generate `pipeline.pkl` fresh in the deployment environment).
2. Go to https://share.streamlit.io and create a new app.
3. Set the entry file to `app.py`.
4. Set the requirements file to `requirements.txt` (the lightweight one — the training-only extras in `requirements_local.txt` aren't needed to *serve* predictions).
5. Go to **Settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   SUPABASE_URL = "your_supabase_project_url"
   SUPABASE_KEY = "your_supabase_api_key"
   ```
6. Deploy — the app reads secrets automatically via `st.secrets` (and `python-dotenv` for local `.env` fallback).

> **Note:** You must also upload or generate `pipeline.pkl` in the deployment environment. The easiest approach is to run `python modeling_pipeline.py` once locally and commit the generated `pipeline.pkl` file to the repo.

---

## How to Use the App

### Dashboard
The landing page. Shows system status, model performance highlights (ROC-AUC, F1, Precision, Recall), model comparison chart (CV vs Test ROC-AUC per model, from `model_comparison.csv`), and the top risk drivers ranked by feature importance.

### Prediction → Batch
Upload a CSV file where each row is one business (or use `Data/sample_batch_demo.csv` to try it instantly). The file must contain these exact columns:

```
credit_amount, monthly_income_avg, total_deposits_3m, revenue_volatility_3m,
request_ratio, dti_monthly, nsf_count_3m, negative_days_3m,
owner_percentage, owner_credit_score, business_age_months
```

Columns can be in **any order** — the app reorders them internally to match the training schema. The app returns:
- KPI summary cards (total SMEs, high-risk count, avg risk score, avg credit score)
- Risk distribution donut chart, average credit by risk level, risk score histogram
- Top 5 highest-risk businesses
- Portfolio-level SHAP feature contribution chart
- A downloadable scored CSV with added columns: `prediction`, `risk_score`, `confidence`, `risk_label`

### Prediction → Single Business
Fill in the financial fields for one business:

| Field | Description |
|---|---|
| Credit Amount (EGP) | Loan amount requested |
| Monthly Income Avg | Average monthly revenue |
| Total Deposits (3M) | Total deposits over last 3 months |
| Revenue Volatility | 0 = stable, 1 = very volatile |
| Debt-to-Income Ratio | Monthly debt / monthly income |
| Owner Credit Score | Personal credit score (300–850) |
| NSF Count (3M) | Non-sufficient funds events |
| Negative Balance Days | Days with negative account balance |
| Owner Percentage | Owner's equity share (%) |
| Request Ratio | New funds / existing commitment |
| Business Age (Months) | How long the business has operated |

Click **Run AI Analysis** to get:
- Risk score gauge (0–100%) + confidence breakdown
- Risk label with bilingual explanation
- AI Insights (rule-based bilingual signal extraction)
- Risk factor breakdown chart + 6-month financial trend simulation
- Strategic recommendations (rule-based, bilingual)
- Gemini AI strategic advisory (if key configured)
- Downloadable bilingual PDF report
- Prediction is saved to Supabase prediction history automatically

### Analytics
Visualizes the training dataset (from Supabase if configured, else local CSV fallback) with:
- Risk class distribution (bar + pie)
- Feature importance (top 10 by importance score)
- SHAP-based model explainability (dataset-level, top 12 features)
- NSF count distribution by risk class
- DTI vs Monthly Income scatter plot
- Feature correlation heatmap
- Model comparison chart (CV vs Test ROC-AUC across all 7 models)
- Model performance metrics chart (final calibrated model)

### Reports
- View and download all PDF reports generated during the session
- Browse full prediction history (from Supabase) with timestamps
- Download history as CSV
- Clear history (clears both the Supabase table and the cached view)

---

## Model Performance (Final Calibrated Model)

| Metric | Score |
|---|---|
| ROC AUC | 98.04% |
| Accuracy | 92.11% |
| Precision | 92.97% |
| Recall | 90.03% |
| F1 Score | 91.48% |

Best model: **Gradient Boosting** (selected by 10-fold cross-validated ROC-AUC, tuned via RandomizedSearchCV, calibrated with isotonic regression)

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| `credit_amount` | float | Loan amount requested (EGP) |
| `monthly_income_avg` | float | Average monthly revenue |
| `total_deposits_3m` | float | Total deposits over 3 months |
| `revenue_volatility_3m` | float | Revenue standard deviation ratio (0–1) |
| `request_ratio` | float | New credit / existing commitment |
| `dti_monthly` | float | Debt-to-income ratio (0–1) |
| `nsf_count_3m` | int | NSF (bounced payment) events in 3 months |
| `negative_days_3m` | int | Days with negative bank balance |
| `owner_percentage` | float | Owner equity share (0–100) |
| `owner_credit_score` | int | Owner personal credit score (300–850) |
| `business_age_months` | int | Months since business registration |

The pipeline automatically derives **12 additional engineered features** internally via `SMEFeatureEngineer` (credit-income ratio, deposit coverage, revenue stability, owner reliability, liquidity stress ratio, and more) — this transformer lives *inside* the sklearn pipeline to avoid data leakage.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, custom CSS (bilingual + RTL) |
| ML Framework | scikit-learn, XGBoost |
| Hyperparameter Tuning | RandomizedSearchCV (10-fold Stratified CV) |
| Experiment Tracking | MLflow (SQLite-backed tracking store + Model Registry) |
| Explainability | SHAP (TreeExplainer) |
| Database | Supabase (PostgreSQL) — prediction history + optional dataset sync |
| AI Advisory | Google Gemini API |
| PDF Generation | ReportLab (bilingual) |
| Language | Python 3.10+ |

---

## Built For

**DEPI — Digital Egypt Pioneers Initiative**
Final graduation project submission.

---

## License

This project is for educational and research purposes.
