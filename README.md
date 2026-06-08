# SME Risk Intelligence Platform

> An AI-powered financial risk assessment system for Small & Medium Enterprises (SMEs), built with Python, Streamlit, and machine learning.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red) ![ML](https://img.shields.io/badge/Model-HistGradientBoosting-green) ![ROC AUC](https://img.shields.io/badge/ROC%20AUC-97.95%25-brightgreen)

---

## What This Project Does

The SME Risk Intelligence Platform helps banks and credit analysts make better lending decisions. Given a set of financial indicators for a business, the system:

- Predicts the **risk level** (Low / Medium / High) with a probability score
- Explains **why** the model made that prediction (SHAP-based explainability)
- Generates **strategic recommendations** using rule-based logic and Google Gemini AI
- Produces a **downloadable PDF report** with a full risk assessment
- Supports **batch scoring** — upload a CSV of multiple businesses and score all at once
- Provides an **analytics dashboard** with model performance, feature importance, and portfolio insights
- Fully **bilingual** — English and Arabic (with RTL layout support)

---

## Project Structure

```
SME-Risk-Intelligence/
│
├── app.py                          # Main Streamlit application entry point
├── modeling_pipeline.py            # Model training & comparison script
├── requirements.txt                # Python dependencies
│
├── Data/
│   └── SMEs_Data.csv               # Training dataset (required to retrain)
│
├── utils/
│   ├── model_loader.py             # Pipeline loading & prediction logic
│   ├── helper.py                   # Feature engineering transformer & utilities
│   ├── history_manager.py          # Prediction history (CSV-based)
│   └── dashboard_stats.py          # Dashboard statistics helper
│
├── components/
│   ├── charts.py                   # Gauge, risk breakdown, trend charts
│   ├── analytics.py                # Full analytics dashboard charts
│   ├── recommendation_engine.py    # Rule-based bilingual recommendations
│   ├── ai_recommendation_engine.py # Gemini AI advisory generator
│   ├── ai_insights.py              # AI signal extraction
│   ├── report_generator.py         # PDF report builder (ReportLab)
│   ├── risk_badge.py               # Risk label badge component
│   └── ui_cards.py                 # Metric card components
│
├── assets/
│   ├── styles.css                  # Global CSS (themes, cards, buttons)
│   ├── dark mode_Logo.png          # Logo for dark theme
│   ├── light mode_Logo.png         # Logo for light theme
│   └── animation.json              # Lottie animation file
│
├── pipeline.pkl                    # Trained model pipeline (generated)
├── model_comparison.csv            # Model comparison results (generated)
├── model_comparison_results.json   # Final metrics (generated)
└── prediction_history.csv          # Prediction log (generated at runtime)
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sme-risk-intelligence.git
cd sme-risk-intelligence
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

> **Required before first run.** This generates `pipeline.pkl`, `model_comparison.csv`, and `model_comparison_results.json`.

```bash
python modeling_pipeline.py
```

The script will:
- Load `Data/SMEs_Data.csv`
- Train and compare 7 ML models (XGBoost, HistGradientBoosting, RandomForest, etc.)
- Select the best model by cross-validated ROC-AUC
- Save the calibrated pipeline to `pipeline.pkl`

### 4. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Enabling Gemini AI Recommendations (Optional)

The AI advisory feature uses Google Gemini. Without a key, the rest of the app works normally — only the AI advisory section will show a fallback message.

**Local setup:**

1. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

2. Get a free API key at: https://aistudio.google.com/app/apikey

**Never commit your `.env` file.** It is already listed in `.gitignore`.

---

## Deployment on Streamlit Cloud

1. Push your project to a GitHub repository (without `.env` or `pipeline.pkl`)
2. Go to https://share.streamlit.io and create a new app
3. Set the entry file to `app.py`
4. Go to **Settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```
5. Deploy — the app reads secrets automatically via `python-dotenv`

> **Note:** You must also upload or generate `pipeline.pkl` in the deployment environment. The easiest approach is to run `modeling_pipeline.py` once and commit the generated `.pkl` file, or use Streamlit Cloud's file system.

---

## How to Use the App

### Dashboard
The landing page. Shows system status, model performance highlights (ROC-AUC, F1, Precision, Recall), model comparison chart, and the top 6 risk drivers ranked by feature importance.

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
- Risk score gauge (0–100%)
- Risk label with confidence level
- AI explanation
- Risk factor breakdown chart
- 6-month financial trend simulation
- Strategic recommendations
- Gemini AI advisory (if key configured)
- Downloadable PDF report

### Prediction → Batch
Upload a CSV file where each row is one business. The file must contain these exact columns:

```
credit_amount, monthly_income_avg, total_deposits_3m, revenue_volatility_3m,
request_ratio, dti_monthly, nsf_count_3m, negative_days_3m,
owner_percentage, owner_credit_score, business_age_months
```

The app returns a scored CSV with added columns: `prediction`, `risk_score`, `confidence`, `risk_label`.

### Analytics
Visualizes the training dataset with:
- Risk class distribution (bar + pie)
- Feature importance (XGBoost)
- SHAP-based model explainability
- NSF count distribution by risk class
- DTI vs Monthly Income scatter plot
- Feature correlation heatmap
- Model performance comparison

### Reports
- View and download all PDF reports generated during the session
- Browse full prediction history with timestamps
- Download history as CSV
- Clear history

---

## Model Performance

| Metric | Score |
|---|---|
| ROC AUC | 97.95% |
| Accuracy | 92.01% |
| Precision | 93.45% |
| Recall | 89.27% |
| F1 Score | 91.31% |

Best model: **Hist Gradient Boosting** (selected by 10-fold cross-validated ROC-AUC)

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

The pipeline automatically derives 12 additional engineered features internally.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, custom CSS |
| ML Framework | scikit-learn, XGBoost |
| Explainability | SHAP |
| AI Advisory | Google Gemini API |
| PDF Generation | ReportLab |
| Language | Python 3.10+ |

---

## Built For

**DEPI — Digital Egypt Pioneers Initiative**  
Final graduation project submission.

---

## License

This project is for educational and research purposes.
