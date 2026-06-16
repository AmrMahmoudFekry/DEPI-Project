
import streamlit as st
import pandas as pd
import json
import os
import base64
from datetime import datetime
from pathlib import Path
import shap
import numpy as np

from utils.model_loader import (
    load_pipeline,
    predict_risk,
    predict_batch_risk,
    TRAINING_COLUMN_ORDER
)

from utils.helper import prepare_input

from utils.history_manager import (
    save_prediction,
    load_history
)
from components.ui_cards import metric_card

from components.charts import (
    create_gauge_chart,
    create_risk_breakdown_chart,
    create_trend_chart
)
from components.analytics import (
    risk_distribution,
    correlation_heatmap,
    financial_overview_chart,
    risk_segments_chart,
    feature_importance_chart,
    nsf_distribution_chart,
    shap_summary_chart,
    model_comparison_chart,
    model_performance_chart
)
from components.recommendation_engine import generate_recommendations
from components.ai_insights import generate_ai_insights
from components.ai_recommendation_engine import generate_ai_recommendations
from components.risk_badge import risk_badge
from components.dashboard_stats import get_dashboard_stats
from components.report_generator import generate_pdf


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SME Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "report" not in st.session_state:
    st.session_state.report = None

if "saved_reports" not in st.session_state:
    st.session_state.saved_reports = []

if "lang" not in st.session_state:
    st.session_state.lang = "en"


# =========================================================
# LOAD CSS
# =========================================================
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =========================================================
# THEME SYSTEM
# =========================================================
def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()


theme = st.session_state.theme
lang  = st.session_state.lang

LANG_LABELS = {
    "en": "English",
    "ar": "العربية"
}

# =========================================================
# TRANSLATIONS
# =========================================================
TRANSLATIONS = {
    "en": {
        "Language": "Language",
        "Navigation": "Navigation",
        "Dashboard": "Dashboard",
        "Prediction": "Prediction",
        "Analytics": "Analytics",
        "Reports": "Reports",
        "System Overview": "System Overview",
        "Model Snapshot": "Model Snapshot",
        "Dataset Summary": "Dataset Summary",
        "AI Status": "AI Status",
        "Platform": "Platform",
        "Theme Mode": "Theme Mode",
        "Predictions": "Predictions",
        "Pipeline": "Pipeline",
        "Avg Confidence": "Avg Confidence",
        "Last Update": "Last Update",
        "Rows": "Rows",
        "Features": "Features",
        "Low Risk": "Low Risk",
        "High Risk": "High Risk",
        "Medium Risk": "Medium Risk",
        "Model Performance Highlights": "Model Performance Highlights",
        "Model Comparison": "Model Comparison",
        "Key Risk Drivers": "Key Risk Drivers",
        "Risk Prediction Engine": "Risk Prediction Engine",
        "Batch Prediction": "Batch Prediction",
        "Single Business Prediction": "Single Business Prediction",
        "Upload a portfolio CSV with one business per row and get a batch risk report. The CSV must contain the same base input features used by the model.": "Upload a portfolio CSV with one business per row and get a batch risk report. The CSV must contain the same base input features used by the model.",
        "Required Columns": "Required Columns",
        "Fill the most important inputs below. Advanced fields are hidden unless you need finer control.": "Fill the most important inputs below. Advanced fields are hidden unless you need finer control.",
        "Run AI Analysis 🚀": "Run AI Analysis 🚀",
        "Analytics Dashboard": "Analytics Dashboard",
        "Target Variable Analysis": "Target Variable Analysis",
        "Feature Analysis": "Feature Analysis",
        "Financial Relationships": "Financial Relationships",
        "Feature Correlation Heatmap": "Feature Correlation Heatmap",
        "Model Explainability": "Model Explainability",
        "Reports Center": "Reports Center",
        "Generated AI Reports": "Generated AI Reports",
        "AI-powered SME Risk Intelligence Platform": "AI-powered SME Risk Intelligence Platform",
        "Dashboard Overview": "Dashboard Overview",
        "AI-Powered Financial Risk Assessment": "AI-Powered Financial Risk Assessment",
        "Upload portfolio CSV": "Upload portfolio CSV",
        "Advanced business details": "Advanced business details",
        "Required columns: credit_amount, monthly_income_avg, total_deposits_3m, revenue_volatility_3m, request_ratio, dti_monthly, nsf_count_3m, negative_days_3m, owner_percentage, owner_credit_score, business_age_months": "Required columns: credit_amount, monthly_income_avg, total_deposits_3m, revenue_volatility_3m, request_ratio, dti_monthly, nsf_count_3m, negative_days_3m, owner_percentage, owner_credit_score, business_age_months",
        "credit_amount, monthly_income_avg, total_deposits_3m,\nrevenue_volatility_3m, request_ratio, dti_monthly,\nnsf_count_3m, negative_days_3m, owner_percentage,\nowner_credit_score, business_age_months": "credit_amount, monthly_income_avg, total_deposits_3m,\nrevenue_volatility_3m, request_ratio, dti_monthly,\nnsf_count_3m, negative_days_3m, owner_percentage,\nowner_credit_score, business_age_months",
        "Batch Prediction Summary": "Batch Prediction Summary",
        "Sample Batch Output": "Sample Batch Output",
        "AI Explanation": "AI Explanation",
        "AI Insights": "AI Insights",
        "Strategic Recommendations": "Strategic Recommendations",
        "AI Strategic Advisory": "AI Strategic Advisory",
        "Running AI Analysis...": "Running AI Analysis...",
        "Generating AI financial advisory...": "Generating AI financial advisory...",
        "✅ AI Analysis Completed": "✅ AI Analysis Completed",
        "📥 Download Full AI Report (PDF)": "📥 Download Full AI Report (PDF)",
        "📥 Download": "📥 Download",
        "Dataset Preview": "Dataset Preview",
        "No PDF reports generated yet. Run an analysis in the **Prediction** page to generate reports.": "No PDF reports generated yet. Run an analysis in the **Prediction** page to generate reports.",
        "Prediction History": "Prediction History",
        "✅ {count} prediction(s) found in history": "✅ {count} prediction(s) found in history",
        "📥 Download Prediction History (CSV)": "📥 Download Prediction History (CSV)",
        "🗑️ Clear Prediction History": "🗑️ Clear Prediction History",
        "✅ Prediction history cleared successfully": "✅ Prediction history cleared successfully",
        "No prediction history available yet.": "No prediction history available yet.",
        "Prediction Engine Active": "Prediction Engine Active",
        "Analytics System Online": "Analytics System Online",
        "Monitoring Financial Risk": "Monitoring Financial Risk",
        "Compare candidate models from the latest training run and highlight the winning estimator.": "Compare candidate models from the latest training run and highlight the winning estimator.",
        "Understand what drives the prediction with SHAP-based feature contributions.": "Understand what drives the prediction with SHAP-based feature contributions.",
        "Why SHAP is included": "Why SHAP is included",
        "SHAP explains the model's decision path. Gemini recommendations are AI-generated action items based on the same risk result.": "SHAP explains the model's decision path. Gemini recommendations are AI-generated action items based on the same risk result.",
        "SHAP works on the prediction model, while Gemini works on recommendation text.": "SHAP works on the prediction model, while Gemini works on recommendation text.",
        "SHAP shows why the model chose High Risk or Low Risk for this business.": "SHAP shows why the model chose High Risk or Low Risk for this business.",
        "Gemini translates the risk result into practical business actions.": "Gemini translates the risk result into practical business actions.",
        "Prediction Confidence": "Prediction Confidence",
        "Confidence": "Confidence",
        "Decision margin": "Decision margin",
        "above random chance": "above random chance",
        "The model is": "The model is",
        "certain": "certain",
        "in this prediction.": "in this prediction.",
        "High Confidence": "High Confidence",
        "Medium Confidence": "Medium Confidence",
        "Low Confidence": "Low Confidence",
        # Input labels
        "Credit Amount (EGP)": "Credit Amount (EGP)",
        "Average Monthly Income (EGP)": "Average Monthly Income (EGP)",
        "Total Deposits (3M)": "Total Deposits (3M)",
        "Revenue Volatility (0-1)": "Revenue Volatility (0-1)",
        "Debt-to-Income Ratio": "Debt-to-Income Ratio",
        "Owner Credit Score": "Owner Credit Score",
        "NSF Count (3M)": "NSF Count (3M)",
        "Negative Balance Days (3M)": "Negative Balance Days (3M)",
        "Owner Percentage (%)": "Owner Percentage (%)",
        "Request Ratio": "Request Ratio",
        "Business Age (Months)": "Business Age (Months)",
        # Help texts
        "The current requested amount or outstanding exposure.": "The current requested amount or outstanding exposure.",
        "The business's average monthly revenue / cash inflow.": "The business's average monthly revenue / cash inflow.",
        "Total cash deposited over the last 3 months.": "Total cash deposited over the last 3 months.",
        "Higher values mean more unstable revenue.": "Higher values mean more unstable revenue.",
        "Monthly debt payments divided by monthly income.": "Monthly debt payments divided by monthly income.",
        "Personal credit score of the business owner.": "Personal credit score of the business owner.",
        "Number of non-sufficient-funds events in the last 3 months.": "Number of non-sufficient-funds events in the last 3 months.",
        "Days with a negative balance in the last 3 months.": "Days with a negative balance in the last 3 months.",
        "Equity share controlled by the business owner.": "Equity share controlled by the business owner.",
        "Ratio of new funds requested to existing loan commitment.": "Ratio of new funds requested to existing loan commitment.",
        "How long the company has been operating.": "How long the company has been operating.",
    },
    "ar": {
        "Language": "اللغة",
        "Navigation": "التنقل",
        "Dashboard": "اللوحة الرئيسية",
        "Prediction": "التنبؤ",
        "Analytics": "التحليلات",
        "Reports": "التقارير",
        "System Overview": "نظرة عامة على النظام",
        "Model Snapshot": "لمحة عن النموذج",
        "Dataset Summary": "ملخص مجموعة البيانات",
        "AI Status": "حالة الذكاء الاصطناعي",
        "Platform": "المنصة",
        "Theme Mode": "نمط العرض",
        "Predictions": "التنبؤات",
        "Pipeline": "سير العمل",
        "Avg Confidence": "متوسط الثقة",
        "Last Update": "آخر تحديث",
        "Rows": "الصفوف",
        "Features": "المتغيرات",
        "Low Risk": "مخاطر منخفضة",
        "High Risk": "مخاطر عالية",
        "Medium Risk": "مخاطر متوسطة",
        "Best Model": "أفضل نموذج",
        "CV ROC AUC": "متوسط ROC AUC",
        "Test ROC AUC": "ROC AUC على الاختبار",
        "Total Predictions": "إجمالي التنبؤات",
        "High Risk Cases": "حالات المخاطر العالية",
        "Model Performance Highlights": "أبرز أداء النموذج",
        "Model Comparison": "مقارنة النماذج",
        "Key Risk Drivers": "عوامل الخطر الرئيسية",
        "Risk Prediction Engine": "محرك التنبؤ بالمخاطر",
        "Batch Prediction": "التنبؤ بالجملة",
        "Single Business Prediction": "تنبؤ عمل فردي",
        "Upload a portfolio CSV with one business per row and get a batch risk report. The CSV must contain the same base input features used by the model.": "حمّل ملف CSV لمحفظة بها كل شركة في سطر واحد واحصل على تقرير مخاطر جماعي. يجب أن يحتوي الملف على نفس المتغيرات الأساسية المستخدمة في النموذج.",
        "Required Columns": "الأعمدة المطلوبة",
        "Fill the most important inputs below. Advanced fields are hidden unless you need finer control.": "املأ أهم المدخلات أدناه. الحقول المتقدمة مخفية إلا إذا كنت تحتاج تحكمًا أدق.",
        "Run AI Analysis 🚀": "تشغيل تحليل الذكاء الاصطناعي 🚀",
        "Analytics Dashboard": "لوحة التحليلات",
        "Target Variable Analysis": "تحليل المتغير الهدف",
        "Feature Analysis": "تحليل المتغيرات",
        "Financial Relationships": "العلاقات المالية",
        "Feature Correlation Heatmap": "خريطة الارتباط بين المتغيرات",
        "Model Explainability": "تفسير النموذج",
        "Reports Center": "مركز التقارير",
        "Generated AI Reports": "تقارير الذكاء الاصطناعي المولدة",
        "AI-powered SME Risk Intelligence Platform": "منصة ذكاء مخاطر المشروعات الصغيرة والمتوسطة",
        "Dashboard Overview": "نظرة عامة على اللوحة",
        "AI-Powered Financial Risk Assessment": "تقييم مخاطر مالي مدعوم بالذكاء الاصطناعي",
        "Upload portfolio CSV": "تحميل ملف CSV للمحفظة",
        "Advanced business details": "تفاصيل الأعمال المتقدمة",
        "Required columns: credit_amount, monthly_income_avg, total_deposits_3m, revenue_volatility_3m, request_ratio, dti_monthly, nsf_count_3m, negative_days_3m, owner_percentage, owner_credit_score, business_age_months": "الأعمدة المطلوبة: credit_amount, monthly_income_avg, total_deposits_3m, revenue_volatility_3m, request_ratio, dti_monthly, nsf_count_3m, negative_days_3m, owner_percentage, owner_credit_score, business_age_months",
        "credit_amount, monthly_income_avg, total_deposits_3m,\nrevenue_volatility_3m, request_ratio, dti_monthly,\nnsf_count_3m, negative_days_3m, owner_percentage,\nowner_credit_score, business_age_months": "credit_amount, monthly_income_avg, total_deposits_3m,\nrevenue_volatility_3m, request_ratio, dti_monthly,\nnsf_count_3m, negative_days_3m, owner_percentage,\nowner_credit_score, business_age_months",
        "Batch Prediction Summary": "ملخص التنبؤ بالجملة",
        "Sample Batch Output": "عينة من نتائج التنبؤ بالجملة",
        "AI Explanation": "شرح الذكاء الاصطناعي",
        "AI Insights": "تحليلات الذكاء الاصطناعي",
        "Strategic Recommendations": "توصيات استراتيجية",
        "AI Strategic Advisory": "استشارة استراتيجية بالذكاء الاصطناعي",
        "Running AI Analysis...": "جارٍ تشغيل تحليل الذكاء الاصطناعي...",
        "Generating AI financial advisory...": "جارٍ إنشاء استشارة مالية بالذكاء الاصطناعي...",
        "✅ AI Analysis Completed": "✅ اكتمل تحليل الذكاء الاصطناعي",
        "📥 Download Full AI Report (PDF)": "📥 تنزيل تقرير الذكاء الاصطناعي الكامل (PDF)",
        "📥 Download": "📥 تنزيل",
        "Dataset Preview": "معاينة مجموعة البيانات",
        "No PDF reports generated yet. Run an analysis in the **Prediction** page to generate reports.": "لم يتم إنشاء تقارير PDF بعد. شغّل تحليلًا في صفحة التنبؤ لإنشاء التقارير.",
        "Prediction History": "سجل التنبؤات",
        "✅ {count} prediction(s) found in history": "✅ تم العثور على {count} تنبؤ(ات) في السجل",
        "📥 Download Prediction History (CSV)": "📥 تنزيل سجل التنبؤات (CSV)",
        "🗑️ Clear Prediction History": "🗑️ مسح سجل التنبؤات",
        "✅ Prediction history cleared successfully": "✅ تم مسح سجل التنبؤات بنجاح",
        "No prediction history available yet.": "لا يوجد سجل تنبؤات حتى الآن.",
        "Prediction Engine Active": "محرك التنبؤ نشط",
        "Analytics System Online": "نظام التحليلات متصل",
        "Monitoring Financial Risk": "مراقبة المخاطر المالية جارية",
        "Compare candidate models from the latest training run and highlight the winning estimator.": "مقارنة النماذج المرشحة من أحدث تشغيل تدريبي وإبراز النموذج الفائز.",
        "Understand what drives the prediction with SHAP-based feature contributions.": "افهم ما الذي يحرك التنبؤ من خلال مساهمات المتغيرات المستندة إلى SHAP.",
        "Why SHAP is included": "لماذا أضفنا SHAP",
        "SHAP explains the model's decision path. Gemini recommendations are AI-generated action items based on the same risk result.": "SHAP يشرح مسار قرار نموذج التنبؤ نفسه، بينما توصيات Gemini هي عناصر عمل ذكية مستندة إلى نفس نتيجة المخاطرة.",
        "SHAP works on the prediction model, while Gemini works on recommendation text.": "SHAP يعمل داخل نموذج التنبؤ، بينما Gemini ينتج نصائح توصيات.",
        "SHAP shows why the model chose High Risk or Low Risk for this business.": "SHAP يوضح لماذا اختار النموذج مخاطر عالية أو منخفضة لهذه الشركة.",
        "Gemini translates the risk result into practical business actions.": "Gemini يحول نتيجة المخاطرة إلى إجراءات عملية للشركة.",
        "Prediction Confidence": "ثقة التنبؤ",
        "Confidence": "الثقة",
        "Decision margin": "هامش القرار",
        "above random chance": "فوق الاحتمال العشوائي",
        "The model is": "النموذج",
        "certain": "متأكد",
        "in this prediction.": "في هذا التنبؤ.",
        "High Confidence": "ثقة عالية",
        "Medium Confidence": "ثقة متوسطة",
        "Low Confidence": "ثقة منخفضة",
        # Input labels
        "Credit Amount (EGP)": "مبلغ الائتمان (جنيه)",
        "Average Monthly Income (EGP)": "متوسط الدخل الشهري (جنيه)",
        "Total Deposits (3M)": "إجمالي الإيداعات (3 أشهر)",
        "Revenue Volatility (0-1)": "تذبذب الإيرادات (0-1)",
        "Debt-to-Income Ratio": "نسبة الدين إلى الدخل",
        "Owner Credit Score": "الدرجة الائتمانية للمالك",
        "NSF Count (3M)": "عدد حالات NSF (3 أشهر)",
        "Negative Balance Days (3M)": "أيام الرصيد السلبي (3 أشهر)",
        "Owner Percentage (%)": "نسبة المالك (%)",
        "Request Ratio": "نسبة الطلب",
        "Business Age (Months)": "عمر الشركة (شهر)",
        # Help texts
        "The current requested amount or outstanding exposure.": "المبلغ المطلوب الحالي أو التعرض القائم.",
        "The business's average monthly revenue / cash inflow.": "متوسط الإيرادات الشهرية للشركة / التدفق النقدي.",
        "Total cash deposited over the last 3 months.": "إجمالي النقد المودع خلال الأشهر الثلاثة الماضية.",
        "Higher values mean more unstable revenue.": "القيم الأعلى تعني إيرادات أقل استقرارًا.",
        "Monthly debt payments divided by monthly income.": "المدفوعات الشهرية للديون مقسومة على الدخل الشهري.",
        "Personal credit score of the business owner.": "الدرجة الائتمانية الشخصية لمالك الشركة.",
        "Number of non-sufficient-funds events in the last 3 months.": "عدد حالات عدم كفاية الأموال في الأشهر الثلاثة الماضية.",
        "Days with a negative balance in the last 3 months.": "أيام الرصيد السلبي في الأشهر الثلاثة الماضية.",
        "Equity share controlled by the business owner.": "نسبة الأسهم التي يسيطر عليها مالك الشركة.",
        "Ratio of new funds requested to existing loan commitment.": "نسبة الأموال الجديدة المطلوبة إلى الالتزام القائم بالقرض.",
        "How long the company has been operating.": "مدة تشغيل الشركة.",
    }
}


def t(key):
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


# =========================================================
# DYNAMIC COLORS & GRADIENTS FIX
# =========================================================
if theme == "dark":
    bg            = "#0B1220"
    # Sidebar gradient for Dark Mode exactly as specified in CSS
    sidebar_bg    = "linear-gradient(180deg, rgba(15,23,42,0.95) 0%, rgba(17,24,39,0.98) 100%)"
    card          = "#111827"
    text          = "#F8FAFC"
    logo_path     = "assets/dark mode_Logo.png"
else:
    bg            = "#F8FAFC"
    # Sidebar gradient for Light Mode exactly as specified in CSS
    sidebar_bg    = "linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%)"
    card          = "#FFFFFF"
    text          = "#0F172A"
    logo_path     = "assets/light mode_Logo.png"

ai_box_bg    = "rgba(17,24,39,0.95)" if theme == "dark" else "#FFFFFF"
ai_box_bdr   = "rgba(255,255,255,0.06)" if theme == "dark" else "#CBD5E1"
ai_box_text  = "#F8FAFC" if theme == "dark" else "#0F172A"
badge_text   = "#CBD5E1" if theme == "dark" else "#334155"

# =========================================================
# RTL — safe, scoped approach (NO global input overrides)
# =========================================================
rtl_style = ""
if lang == "ar":
    rtl_style = """
    /* RTL layout for Arabic — scoped to text/layout only */
    .stApp {
        direction: rtl;
    }
    /* Sidebar RTL */
    section[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    /* Main content RTL */
    .main .block-container {
        direction: rtl;
        text-align: right;
    }
    /* Headers and text */
    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp label,
    .stApp .stMarkdown {
        text-align: right;
    }
    /* Markdown content */
    .stMarkdown p, .stMarkdown li, .stMarkdown h1,
    .stMarkdown h2, .stMarkdown h3 {
        direction: rtl;
        text-align: right;
    }
    /* Fix: keep Streamlit widget internals LTR so they render correctly */
    .stNumberInput input,
    .stTextInput input,
    .stSlider,
    .stSlider * {
        direction: ltr !important;
    }
    /* Align number input label right and keep help icon placement consistent */
    .stNumberInput label,
    .stTextInput label,
    .stSlider label,
    .stSelectbox label,
    .stFileUploader label {
        direction: rtl !important;
        text-align: right !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        flex-wrap: nowrap !important;
        gap: 0.45rem !important;
        white-space: nowrap !important;
    }
    .stNumberInput label button,
    .stTextInput label button,
    .stSlider label button,
    .stSelectbox label button,
    .stFileUploader label button {
        order: -1 !important;
        margin: 0 !important;
        padding: 0 !important;
        align-self: center !important;
    }
    .stNumberInput label > div,
    .stTextInput label > div,
    .stSlider label > div,
    .stSelectbox label > div,
    .stFileUploader label > div {
        order: 0 !important;
    }
    /* Fix metric cards text alignment */
    [data-testid="metric-container"] {
        direction: rtl;
        text-align: right;
    }
    /* Expander header */
    .streamlit-expanderHeader {
        direction: rtl;
        text-align: right;
    }
    /* Columns stay LTR (Streamlit grid) but content RTL */
    [data-testid="column"] {
        direction: rtl;
    }
    """

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
    }}
    h1,h2,h3,h4,h5,h6,p,span,label {{
        color:{text} !important;
    }}
    .ai-box {{
        background: {ai_box_bg} !important;
        border: 1px solid {ai_box_bdr} !important;
        color: {ai_box_text} !important;
    }}
    .ai-box h4, .ai-box p, .ai-box div {{
        color: {ai_box_text} !important;
    }}
    .rec-title {{
        color: {ai_box_text} !important;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 6px;
    }}
    .rec-desc {{
        color: {badge_text} !important;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 12px;
    }}
    .stFileUploader, .stFileUploader > div, .stFileUploader div[data-testid="stFileUploadDropzone"] {{
        background-color: {card} !important;
        color: {text} !important;
        border: 1px solid rgba(148,163,184,0.35) !important;
    }}
    .stFileUploader label, .stFileUploader span, .stFileUploader p {{
        color: {text} !important;
    }}
    .stFileUploader button {{
        background: linear-gradient(90deg, #1E3A8A, #2563EB) !important;
        color: white !important;
    }}
    {rtl_style}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPERS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "Data" / "SMEs_Data.csv"
MODEL_SUMMARY_PATH = ROOT_DIR / "model_comparison_results.json"
PIPELINE_PATH = ROOT_DIR / "pipeline.pkl"


def load_logo_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


def load_model_summary():
    if MODEL_SUMMARY_PATH.exists():
        with open(MODEL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_data_summary():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        low_risk  = int((df["risk_sharp"] == 0).sum()) if "risk_sharp" in df.columns else None
        high_risk = int((df["risk_sharp"] == 1).sum()) if "risk_sharp" in df.columns else None
        return {
            "rows": len(df),
            "features": len(df.columns) - (1 if "risk_sharp" in df.columns else 0),
            "low_risk": low_risk,
            "high_risk": high_risk,
        }
    return None


def format_update_time(path: Path):
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return "N/A"


def confidence_breakdown(confidence, risk_score):
    """Render a visual, meaningful confidence display."""
    prob      = confidence / 100
    certainty = "High" if confidence >= 80 else "Medium" if confidence >= 65 else "Low"

    certainty_color = {
        "High":   "#10B981",
        "Medium": "#F59E0B",
        "Low":    "#EF4444"
    }[certainty]

    margin = round(abs(prob - (1 - prob)) * 100, 1)

    # Translated labels
    conf_label   = t(f"{certainty} Confidence")
    pred_conf    = t("Prediction Confidence")
    dec_margin   = t("Decision margin")
    above_rand   = t("above random chance")
    model_is     = t("The model is")
    cert_word    = t("certain")
    in_pred      = t("in this prediction.")

    st.markdown(
        f"""
        <div style="
            background:{'rgba(17,24,39,0.95)' if theme=='dark' else '#F1F5F9'};
            border-radius:16px;
            padding:20px;
            border:1px solid {'rgba(255,255,255,0.06)' if theme=='dark' else '#CBD5E1'};
            margin-top:12px;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <span style="color:{badge_text};font-size:14px;font-weight:600;">
                    {pred_conf}
                </span>
                <span style="
                    background:{certainty_color};
                    color:white;
                    padding:4px 12px;
                    border-radius:20px;
                    font-size:12px;
                    font-weight:700;
                ">
                    {conf_label}
                </span>
            </div>
            <div style="font-size:36px;font-weight:800;color:{certainty_color};margin-bottom:6px;">
                {confidence}%
            </div>
            <div style="
                background:rgba(100,100,100,0.2);
                border-radius:8px;
                height:10px;
                margin-bottom:10px;
                overflow:hidden;
            ">
                <div style="
                    background:{certainty_color};
                    width:{confidence}%;
                    height:100%;
                    border-radius:8px;
                    transition:width 0.5s ease;
                "></div>
            </div>
            <div style="color:{badge_text};font-size:12px;direction:{'rtl' if lang=='ar' else 'ltr'};">
                {dec_margin}: <b style="color:{certainty_color};">{margin}%</b> {above_rand} ·
                {model_is} <b style="color:{certainty_color};">{certainty.lower()}ly {cert_word}</b>
                {in_pred}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_recommendation_card(rec, idx):
    """Render a styled recommendation card that works in both themes."""
    priority_color = {
        "LOW":      "#10B981",
        "MEDIUM":   "#F59E0B",
        "HIGH":     "#EF4444",
        "CRITICAL": "#DC2626"
    }.get(rec["priority"], "#2563EB")

    priority_icon = {
        "LOW":      "✅",
        "MEDIUM":   "⚠️",
        "HIGH":     "🔴",
        "CRITICAL": "🚨"
    }.get(rec["priority"], "📌")

    box_bg  = "rgba(17,24,39,0.95)" if theme == "dark" else "#FFFFFF"
    box_bdr = "rgba(255,255,255,0.06)" if theme == "dark" else "#CBD5E1"
    txt_clr = "#F8FAFC" if theme == "dark" else "#0F172A"
    sub_clr = "#CBD5E1" if theme == "dark" else "#475569"

    st.markdown(
        f"""
        <div style="
            background:{box_bg};
            border-radius:16px;
            padding:20px 24px;
            border:1px solid {box_bdr};
            border-left:4px solid {priority_color};
            margin-bottom:12px;
            direction:{'rtl' if lang=='ar' else 'ltr'};
        ">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                <div style="font-size:16px;font-weight:700;color:{txt_clr};">
                    {priority_icon} {rec['title']}
                </div>
                <span style="
                    background:{priority_color};
                    color:white;
                    padding:4px 12px;
                    border-radius:20px;
                    font-size:11px;
                    font-weight:700;
                    white-space:nowrap;
                    margin-{'right' if lang=='ar' else 'left'}:12px;
                ">
                    {rec['priority']} PRIORITY
                </span>
            </div>
            <div style="color:{sub_clr};font-size:14px;line-height:1.65;">
                {rec['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIDEBAR RENDER
# =========================================================
with st.sidebar:

    encoded_logo = load_logo_base64(logo_path)

    if encoded_logo:
        st.markdown(
            f'<div class="sidebar-logo-box"><img src="data:image/png;base64,{encoded_logo}" class="logo-img"></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="sidebar-logo-fallback">📊 SME Risk<br>Intelligence</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    lang_choice = st.selectbox(
        t("Language"),
        [LANG_LABELS["en"], LANG_LABELS["ar"]],
        index=0 if st.session_state.lang == "en" else 1,
        key="language_select"
    )
    new_lang = "en" if lang_choice == LANG_LABELS["en"] else "ar"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    page_names  = ["Dashboard", "Prediction", "Analytics", "Reports"]
    page_labels = [t(name) for name in page_names]
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    current_page = st.session_state.page
    selected_label = page_labels[page_names.index(current_page)]
    selected_page = st.selectbox(
        t("Navigation"),
        page_labels,
        index=page_labels.index(selected_label),
        key="page_select"
    )
    page = page_names[page_labels.index(selected_page)]
    st.session_state.page = page

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    model_summary   = load_model_summary()
    data_summary    = load_data_summary()
    pipeline_status = "Ready" if PIPELINE_PATH.exists() else "Missing"
    trained_time    = format_update_time(MODEL_SUMMARY_PATH if MODEL_SUMMARY_PATH.exists() else PIPELINE_PATH)
    best_model      = model_summary["best_model"] if model_summary else "N/A"
    best_model_auc  = model_summary["final_metrics"]["test_roc_auc"] if model_summary else "N/A"
    cv_auc          = (
        model_summary["model_comparison"][best_model]["cv_roc_auc_mean"]
        if model_summary and best_model in model_summary["model_comparison"]
        else "N/A"
    )

    st.markdown(f'<div class="sidebar-section-title">{t("System Overview")}</div>', unsafe_allow_html=True)

    history_df       = load_history()
    total_predictions = len(history_df)
    avg_risk         = round(history_df["risk_score"].mean(), 1) if total_predictions > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric(t("Predictions"), total_predictions)
        st.metric(t("Pipeline"), pipeline_status)
    with col2:
        st.metric(t("Avg Confidence"), f"{avg_risk}%")
        st.metric(t("Last Update"), trained_time)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-section-title">{t("Model Snapshot")}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-info-box" style="direction:{'rtl' if lang=='ar' else 'ltr'};text-align:{'right' if lang=='ar' else 'left'};margin-bottom:16px;">
            <div><strong>{t('Best Model')}:</strong> {best_model}</div>
            <div><strong>{t('Test ROC AUC')}:</strong> {best_model_auc}%</div>
            <div><strong>{t('CV ROC AUC')}:</strong> {cv_auc}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-section-title">{t("Dataset Summary")}</div>', unsafe_allow_html=True)
    data_cols = st.columns(2)
    with data_cols[0]:
        st.metric(t("Rows"),     f"{data_summary['rows']:,}"      if data_summary else "N/A")
        st.metric(t("Low Risk"), f"{data_summary['low_risk']:,}"  if data_summary and data_summary['low_risk'] is not None else "N/A")
    with data_cols[1]:
        st.metric(t("Features"),  data_summary['features']         if data_summary else "N/A")
        st.metric(t("High Risk"), f"{data_summary['high_risk']:,}" if data_summary and data_summary['high_risk'] is not None else "N/A")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-section-title">{t("AI Status")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-status-box success">✅ {t("Prediction Engine Active")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-status-box info">ℹ️ {t("Analytics System Online")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-status-box warning">⚠️ {t("Monitoring Financial Risk")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-section-title">{t("Platform")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-info-box" style="text-align:center;margin-bottom:20px;">{t("AI-powered SME Risk Intelligence Platform")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-section-title">{t("Theme Mode")}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙 Dark"):
            if theme != "dark":
                toggle_theme()
    with col2:
        if st.button("☀️ Light"):
            if theme != "light":
                toggle_theme()


# =========================================================
# DASHBOARD PAGE
# =========================================================
if page == "Dashboard":

    st.markdown(
        f'<div class="main-title">{t("Dashboard Overview")}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="subtitle">{t("AI-powered SME Risk Intelligence Platform")}</div>',
        unsafe_allow_html=True
    )

    stats         = get_dashboard_stats()
    model_summary = load_model_summary()
    final_metrics = model_summary["final_metrics"] if model_summary else {}

    artifact_status = {
        "Dataset":    DATA_PATH.exists(),
        "Pipeline":   PIPELINE_PATH.exists(),
        "Comparison": MODEL_SUMMARY_PATH.exists(),
    }

    st.markdown(f"### 🔧 {t('System Overview')}")
    status_cols = st.columns(3)
    status_labels = [
        ("Dataset", artifact_status["Dataset"]),
        ("Pipeline", artifact_status["Pipeline"]),
        ("Model Summary", artifact_status["Comparison"]),
    ]
    for col, (label, ready) in zip(status_cols, status_labels):
        status = "Ready" if ready else "Missing"
        color  = "#10B981" if ready else "#F59E0B"
        col.markdown(
            f"<div style='padding:18px;border-radius:16px;border:1px solid {color};background:rgba(255,255,255,0.04);'>"
            f"<div style='font-size:14px;color:{color};font-weight:700;'>{label}</div>"
            f"<div style='font-size:20px;color:{color};margin-top:8px;'>{status}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    if not all(artifact_status.values()):
        st.warning(
            "Some project artifacts are missing. Run `python modeling_pipeline.py` to rebuild the model and comparison files before using predictions and analytics."
        )

    st.markdown("---")

    st.markdown(f"### 🏆 {t('Model Performance Highlights')}")
    metric_cols = st.columns(5)
    perf_data = [
        ("ROC AUC",   f"{final_metrics.get('test_roc_auc', 'N/A')}%" if final_metrics else "N/A", "#8B5CF6"),
        ("F1 Score",  f"{final_metrics.get('test_f1', 'N/A')}%"      if final_metrics else "N/A", "#06B6D4"),
        ("Precision", f"{final_metrics.get('test_precision', 'N/A')}%" if final_metrics else "N/A", "#2563EB"),
        ("Recall",    f"{final_metrics.get('test_recall', 'N/A')}%"  if final_metrics else "N/A", "#10B981"),
        ("Best Model", model_summary.get('best_model', 'N/A') if model_summary else "N/A", "#F59E0B"),
    ]
    for col, (label, val, clr) in zip(metric_cols, perf_data):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{'rgba(17,24,39,0.95)' if theme=='dark' else '#F1F5F9'};
                    border-radius:14px;
                    padding:18px;
                    text-align:center;
                    border-top:3px solid {clr};
                    margin-bottom:8px;
                ">
                    <div style="color:{badge_text};font-size:12px;margin-bottom:6px;">{label}</div>
                    <div style="color:{clr};font-size:24px;font-weight:800;">{val}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown(f"### 📈 {t('Model Comparison')}")
    st.markdown(t("Compare candidate models from the latest training run and highlight the winning estimator."))
    model_comparison_chart(theme)

    st.markdown("---")

    st.markdown(f"### 📌 {t('Key Risk Drivers')}")
    feat_cols = st.columns(3)
    drivers = [
        ("NSF Count (3M)",       "28.9%", "Top predictor of financial stress"),
        ("Negative Balance Days","11.3%", "Indicates operational liquidity problems"),
        ("Debt-to-Income Ratio", "7.3%",  "Leverage exposure metric"),
        ("Revenue Volatility",   "6.3%",  "Cash flow stability indicator"),
        ("Monthly Income",       "6.8%",  "Primary repayment capacity signal"),
        ("Owner Credit Score",   "3.6%",  "Personal creditworthiness proxy"),
    ]
    for i, col in enumerate(feat_cols):
        with col:
            pair = [drivers[i * 2]] + ([drivers[i * 2 + 1]] if i * 2 + 1 < len(drivers) else [])
            for item in pair:
                st.markdown(
                    f"""
                    <div style="
                        background:{'rgba(17,24,39,0.95)' if theme=='dark' else '#F1F5F9'};
                        border-radius:12px;
                        padding:14px 16px;
                        margin-bottom:10px;
                        border-left:3px solid #2563EB;
                    ">
                        <div style="font-weight:700;color:{text};font-size:14px;">{item[0]}</div>
                        <div style="color:#2563EB;font-weight:800;font-size:18px;margin:4px 0;">{item[1]}</div>
                        <div style="color:{badge_text};font-size:12px;">{item[2]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")


# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "Prediction":

    st.markdown(
        f'<div class="main-title">{t("Risk Prediction Engine")}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{t("AI-Powered Financial Risk Assessment")}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Batch Prediction ──────────────────────────────────
    st.markdown(f"## {t('Batch Prediction')}")

    st.markdown(
        t(
            "Upload a portfolio CSV with one business per row and get a batch risk report. " + "/n" +
            "The CSV must contain the same base input features used by the model."
        )
    )

    uploaded_file = st.file_uploader(
        t("Upload portfolio CSV"),
        type=["csv"],
        help=t(
            "Required columns: credit_amount, monthly_income_avg, "
            "total_deposits_3m, revenue_volatility_3m, request_ratio, "
            "dti_monthly, nsf_count_3m, negative_days_3m, "
            "owner_percentage, owner_credit_score, business_age_months"
        ),
    )

    st.markdown(f"### {t('Required Columns')}")

    st.markdown(""" credit_amount , monthly_income_avg , total_deposits_3m , revenue_volatility_3m , request_ratio , dti_monthly """)

    st.markdown(""" nsf_count_3m , negative_days_3m , owner_percentage , owner_credit_score , business_age_months """)

    if uploaded_file is not None:
        try:
            import plotly.graph_objects as go

            raw_batch_df = pd.read_csv(uploaded_file)
            pipeline = load_pipeline()
            batch_results = predict_batch_risk(pipeline, raw_batch_df)

            count_by_label = batch_results['risk_label'].value_counts().to_dict()
            low_count = count_by_label.get('LOW RISK', 0)
            med_count = count_by_label.get('MEDIUM RISK', 0)
            high_count = count_by_label.get('HIGH RISK', 0)

            total = len(batch_results)
            avg_score = round(batch_results['risk_score'].mean(), 1)

            avg_credit_score = (
                round(raw_batch_df['owner_credit_score'].mean(), 0)
                if 'owner_credit_score' in raw_batch_df.columns
                else 0
            )

            font_color = "#F8FAFC" if theme == "dark" else "#0F172A"
            bg_color = "rgba(0,0,0,0)" if theme == "dark" else "#FFFFFF"
 
            # ── KPI Cards ────────────────────────────
            st.markdown(f"### 📊 {t('Batch Prediction Summary')}")
            k1, k2, k3, k4 = st.columns(4)
 
            def _kpi(col, label, value, color):
                col.markdown(
                    f"""<div style="
                        background:{'rgba(17,24,39,0.95)' if theme=='dark' else '#F1F5F9'};
                        border-radius:14px;padding:18px;text-align:center;
                        border-top:4px solid {color};margin-bottom:8px;">
                        <div style="color:{badge_text};font-size:12px;margin-bottom:6px;">{label}</div>
                        <div style="color:{color};font-size:28px;font-weight:800;">{value}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
 
            _kpi(k1, "Total SMEs",       total,              "#2563EB")
            _kpi(k2, "High Risk",          high_count,         "#EF4444")
            _kpi(k3, "Avg Risk Score",    f"{avg_score}%",    "#F59E0B")
            _kpi(k4, "Avg Credit Score",  int(avg_credit_score), "#10B981")
 
            st.markdown("---")
 
            # ── Charts Row 1: Donut + Bar ─────────────
            st.markdown("### 📈 Portfolio Risk Visualizations")
            chart_col1, chart_col2 = st.columns(2)
 
            with chart_col1:
                donut_labels = ["Low Risk", "Medium Risk", "High Risk"]
                donut_values = [low_count, med_count, high_count]
                donut_colors = ["#10B981", "#F59E0B", "#EF4444"]
                fig_donut = go.Figure(go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=0.55,
                    marker=dict(colors=donut_colors),
                    textinfo="percent+label",
                    textfont=dict(color=font_color, size=12),
                    pull=[0.03, 0.03, 0.06]
                ))
                fig_donut.update_layout(
                    title="Risk Level Distribution",
                    title_font=dict(color=font_color, size=15),
                    paper_bgcolor=bg_color,
                    font=dict(color=font_color),
                    legend=dict(font=dict(color=font_color)),
                    height=360,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
 
            with chart_col2:
                if 'credit_amount' in batch_results.columns:
                    avg_credit_by_risk = (
                        batch_results.groupby('risk_label')['credit_amount']
                        .mean()
                        .reindex(['LOW RISK', 'MEDIUM RISK', 'HIGH RISK'])
                        .fillna(0)
                        .reset_index()
                    )
                    fig_bar = go.Figure(go.Bar(
                        x=avg_credit_by_risk['risk_label'],
                        y=avg_credit_by_risk['credit_amount'],
                        marker_color=["#10B981", "#F59E0B", "#EF4444"],
                        text=[f"EGP {v:,.0f}" for v in avg_credit_by_risk['credit_amount']],
                        textposition="outside",
                        textfont=dict(color=font_color, size=11)
                    ))
                    fig_bar.update_layout(
                        title="Avg Credit Amount by Risk Level",
                        title_font=dict(color=font_color, size=15),
                        yaxis=dict(title="Avg Credit (EGP)",
                                    tickfont=dict(color=font_color),
                                    gridcolor="rgba(100,100,100,0.15)"),
                        xaxis=dict(tickfont=dict(color=font_color)),
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color,
                        font=dict(color=font_color),
                        height=360,
                        margin=dict(l=20, r=20, t=50, b=20),
                        template="plotly_dark" if theme == "dark" else "plotly_white"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
 
            # ── Charts Row 2: Histogram + Top 5 ──────
            hist_col, top_col = st.columns(2)
 
            with hist_col:
                fig_hist = go.Figure(go.Histogram(
                    x=batch_results['risk_score'],
                    nbinsx=20,
                    marker_color="#2563EB",
                    opacity=0.85
                ))
                fig_hist.add_vline(x=40, line_dash="dash", line_color="#10B981",
                                    annotation_text="Low/Med",
                                    annotation_font_color="#10B981")
                fig_hist.add_vline(x=70, line_dash="dash", line_color="#EF4444",
                                    annotation_text="Med/High",
                                    annotation_font_color="#EF4444")
                fig_hist.update_layout(
                    title="Risk Score Distribution",
                    title_font=dict(color=font_color, size=15),
                    xaxis=dict(title="Risk Score (%)", tickfont=dict(color=font_color)),
                    yaxis=dict(title="Count", tickfont=dict(color=font_color),
                                gridcolor="rgba(100,100,100,0.15)"),
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=font_color),
                    height=360,
                    margin=dict(l=20, r=20, t=50, b=20),
                    template="plotly_dark" if theme == "dark" else "plotly_white"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
 
            with top_col:
                st.markdown(
                    f"""<div style="
                        background:{'rgba(17,24,39,0.95)' if theme=='dark' else '#F1F5F9'};
                        border-radius:14px;padding:16px;
                        border-top:4px solid #EF4444;">
                        <div style="color:#EF4444;font-weight:800;font-size:15px;margin-bottom:12px;">
                            🚨 Top 5 Highest Risk Businesses
                        </div>""",
                    unsafe_allow_html=True
                )
                top5 = batch_results.nlargest(5, 'risk_score')[
                    ['risk_score', 'confidence', 'risk_label']
                ].reset_index()
                for _, row in top5.iterrows():
                    st.markdown(
                        f"""<div style="
                            background:rgba(239,68,68,0.1);
                            border-radius:10px;padding:10px 14px;
                            margin-bottom:8px;
                            border-left:3px solid #EF4444;">
                            <div style="color:{font_color};font-size:13px;">
                                <b>Row {int(row['index'])+1}</b> —
                                Risk Score: <b style="color:#EF4444;">{row['risk_score']}%</b>
                                | Conf: {row['confidence']}%
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)
 
            st.markdown("---")
 
            # ── Batch SHAP ────────────────────────────
            st.markdown("### 🧠 Batch SHAP Summary — Portfolio-Level Feature Impact")
            st.caption("Shows which features drove the risk predictions across all uploaded businesses.")
 
            try:
 
                shap_input     = raw_batch_df.copy()
                missing_cols = [c for c in TRAINING_COLUMN_ORDER if c not in shap_input.columns]
                if missing_cols:
                    st.warning(f"Missing columns for SHAP: {missing_cols}")
                shap_input = shap_input[TRAINING_COLUMN_ORDER].copy()
                pipeline_shap  = load_pipeline()
                prepared       = pipeline_shap.named_steps['feat_eng'].transform(shap_input)
                processed      = pipeline_shap.named_steps['preprocessor'].transform(prepared)
                feature_names = (
                    pipeline_shap
                    .named_steps["preprocessor"]
                    .get_feature_names_out()
                )
 
                classifier = pipeline_shap.named_steps['classifier']
                if hasattr(classifier, 'calibrated_classifiers_'):
                    estimator = classifier.calibrated_classifiers_[0].estimator
                else:
                    estimator = getattr(classifier, 'base_estimator', classifier)
 
                explainer   = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(processed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
 
                mean_shap  = shap_values.mean(axis=0)
                importance = np.abs(mean_shap)
                direction  = np.sign(mean_shap)
 
                shap_df = pd.DataFrame({
                    'feature':    feature_names,
                    'importance': importance,
                    'direction':  direction
                }).sort_values('importance', ascending=False).head(12)
 
                shap_df['color'] = shap_df['direction'].map(
                    {1.0: '#EF4444', -1.0: '#10B981', 0.0: '#6B7280'}
                )
 
                fig_shap = go.Figure(go.Bar(
                    x=shap_df['importance'][::-1],
                    y=shap_df['feature'][::-1],
                    orientation='h',
                    marker_color=list(shap_df['color'][::-1]),
                    text=[f"{v:.3f}" for v in shap_df['importance'][::-1]],
                    textposition='outside',
                    textfont=dict(color=font_color, size=11)
                ))
                fig_shap.update_layout(
                    title='SHAP Feature Contributions (Batch — Top 12)',
                    title_font=dict(color=font_color, size=15),
                    xaxis=dict(title='Mean |SHAP value|',
                                   tickfont=dict(color=font_color)),
                    yaxis=dict(tickfont=dict(color=font_color)),
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=font_color),
                    height=440,
                    margin=dict(l=20, r=20, t=50, b=20),
                    template="plotly_dark" if theme == "dark" else "plotly_white"
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption("🔴 Red = pushes toward High Risk  |  🟢 Green = pushes toward Low Risk")
 
            except Exception as shap_err:
                st.info(f"SHAP analysis unavailable for this batch: {shap_err}")
 
            st.markdown("---")
 
            # ── Table + Download ──────────────────────
            st.markdown(f"### 📋 {t('Sample Batch Output')}")
            st.dataframe(batch_results.head(10), use_container_width=True)
 
            csv_bytes = batch_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Batch Risk Report (CSV)",
                data=csv_bytes,
                file_name="batch_risk_report.csv",
                mime="text/csv"
            )
 
        except KeyError as e:
            st.error(f"Batch CSV validation failed: {str(e)}")
        except Exception as e:
            st.error(f"Unable to run batch prediction: {e}")
 
    st.markdown("---")


    # ── Single Business Prediction ────────────────────────
    st.markdown(f"## {t('Single Business Prediction')}")
    st.markdown(t("Fill the most important inputs below. Advanced fields are hidden unless you need finer control."))

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_amount = st.number_input(
            t("Credit Amount (EGP)"), min_value=0,
            help=t("The current requested amount or outstanding exposure."),
            key="inp_credit_amount"
        )
        monthly_income_avg = st.number_input(
            t("Average Monthly Income (EGP)"), min_value=0,
            help=t("The business's average monthly revenue / cash inflow."),
            key="inp_monthly_income"
        )
        total_deposits_3m = st.number_input(
            t("Total Deposits (3M)"), min_value=0,
            help=t("Total cash deposited over the last 3 months."),
            key="inp_total_deposits"
        )

    with col2:
        revenue_volatility_3m = st.slider(
            t("Revenue Volatility (0-1)"), 0.0, 1.0, 0.3,
            help=t("Higher values mean more unstable revenue."),
            key="inp_rev_volatility"
        )
        dti_monthly = st.slider(
            t("Debt-to-Income Ratio"), 0.0, 1.0, 0.35,
            help=t("Monthly debt payments divided by monthly income."),
            key="inp_dti"
        )
        owner_credit_score = st.slider(
            t("Owner Credit Score"), 300, 850, 650,
            help=t("Personal credit score of the business owner."),
            key="inp_credit_score"
        )

    with col3:
        nsf_count_3m = st.number_input(
            t("NSF Count (3M)"), 0,
            help=t("Number of non-sufficient-funds events in the last 3 months."),
            key="inp_nsf_count"
        )
        negative_days_3m = st.number_input(
            t("Negative Balance Days (3M)"), 0,
            help=t("Days with a negative balance in the last 3 months."),
            key="inp_neg_days"
        )
        owner_percentage = st.slider(
            t("Owner Percentage (%)"), 0.0, 100.0, 50.0,
            help=t("Equity share controlled by the business owner."),
            key="inp_owner_pct"
        )

    with st.expander(t("Advanced business details")):
        request_ratio = st.number_input(
            t("Request Ratio"), min_value=0.0, value=0.1,
            help=t("Ratio of new funds requested to existing loan commitment."),
            key="inp_request_ratio"
        )
        business_age_months = st.number_input(
            t("Business Age (Months)"), min_value=0, value=24,
            help=t("How long the company has been operating."),
            key="inp_business_age"
        )

    st.markdown("---")

    if st.button(t("Run AI Analysis 🚀")):

        with st.spinner(t("Running AI Analysis...")):

            form_data = {
                "credit_amount":         credit_amount,
                "business_age_months":   business_age_months,
                "monthly_income_avg":    monthly_income_avg,
                "total_deposits_3m":     total_deposits_3m,
                "revenue_volatility_3m": revenue_volatility_3m,
                "request_ratio":         request_ratio,
                "dti_monthly":           dti_monthly,
                "nsf_count_3m":          nsf_count_3m,
                "negative_days_3m":      negative_days_3m,
                "owner_percentage":      owner_percentage,
                "owner_credit_score":    owner_credit_score
            }

            try:
                pipeline      = load_pipeline()
                raw_input_df  = pd.DataFrame([form_data])
                result, label = predict_risk(pipeline, raw_input_df)
                input_df      = prepare_input(form_data)

                st.session_state.report = {
                    "result":          result,
                    "input_df":        input_df,
                    "form_data":       form_data,
                    "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "recommendations": None
                }

                save_prediction({
                    "risk_score": result["risk_score"],
                    "confidence": result["confidence"],
                    "risk_label": result["risk_label"]
                })

                st.success(t("✅ AI Analysis Completed"))

            except Exception as e:
                st.error(f"Unable to run single prediction: {e}")
                st.stop()
        st.markdown("---")

        # ── Gauge + Confidence ──
        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(
                create_gauge_chart(result["risk_score"], theme),
                use_container_width=True
            )

        with col2:
            risk_badge(result)
            st.markdown("")
            confidence_breakdown(result["confidence"], result["risk_score"])
            st.markdown(f"#### {t('AI Explanation')}")
            st.write(result["explanation"])

        st.markdown("---")

        # ── AI Insights ──
        summary, insights = generate_ai_insights(result, input_df, lang=lang)
        st.markdown(f"## 🔍 {t('AI Insights')}")
        st.info(summary)
        for item in insights:
            st.write(f"• {item}")

        st.markdown("---")

        # ── Charts ──
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_risk_breakdown_chart(input_df, theme),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                create_trend_chart(input_df, theme),
                use_container_width=True
            )

        st.markdown("---")

        # ── Recommendations ──
        st.markdown(f"## 📋 {t('Strategic Recommendations')}")

        recommendations = generate_recommendations(input_df, result)
        st.session_state.report["recommendations"] = recommendations

        for i, rec in enumerate(recommendations):
            render_recommendation_card(rec, i)

        st.markdown("---")

        # ── AI Strategic Advisory (language-aware) ──
        st.markdown(f"## 🤖 {t('AI Strategic Advisory')}")

        with st.spinner(t("Generating AI financial advisory...")):
            ai_advisory = generate_ai_recommendations(
                result, input_df, recommendations, lang=lang
            )

        box_bg  = "rgba(17,24,39,0.95)" if theme == "dark" else "#FFFFFF"
        box_bdr = "rgba(255,255,255,0.06)" if theme == "dark" else "#CBD5E1"

        st.markdown(
            f"""<div style="
                background:{box_bg};
                border-radius:16px;
                padding:24px;
                border:1px solid {box_bdr};
                direction:{'rtl' if lang=='ar' else 'ltr'};
            ">""",
            unsafe_allow_html=True
        )
        st.markdown(ai_advisory)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── PDF Download ──
        pdf = generate_pdf(
            result["risk_score"],
            result["confidence"],
            result["explanation"],
            result=result,
            input_df=input_df,
            recommendations=recommendations
        )

        report_filename = f"SME_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        st.session_state.saved_reports.append({
            "filename":   report_filename,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_label": result["risk_label"],
            "risk_score": result["risk_score"],
            "data":       pdf
        })

        st.download_button(
            label=t("📥 Download Full AI Report (PDF)"),
            data=pdf,
            file_name=report_filename,
            mime="application/pdf"
        )


# =========================================================
# ANALYTICS PAGE
# =========================================================
elif page == "Analytics":

    st.markdown(
        f'<div class="main-title">{t("Analytics Dashboard")}</div>',
        unsafe_allow_html=True
    )

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        df = pd.read_csv("SMEs_Data.csv")

    st.markdown(f"### 📊 {t('Dataset Summary')}")
    summary_cols = st.columns(4)
    summary_data = [
        ("Total Records",   f"{len(df):,}"),
        ("Features",        str(len(df.columns) - 2)),
        ("Low Risk Cases",  f"{(df['risk_sharp'] == 0).sum():,}"),
        ("High Risk Cases", f"{(df['risk_sharp'] == 1).sum():,}"),
    ]
    for col, (label, val) in zip(summary_cols, summary_data):
        with col:
            metric_card(label, val)

    st.markdown("---")

    st.markdown(f"### 💡 {t('Why this matters')}")
    st.markdown(
        "This dashboard helps decision makers understand SME portfolio risk, identify the strongest risk drivers, "
        "and translate model insights into practical credit actions."
    )
    st.markdown("---")

    st.markdown(f"### 🎯 {t('Target Variable Analysis')}")
    col1, col2 = st.columns(2)
    with col1:
        risk_distribution(df, theme)
    with col2:
        risk_segments_chart(df, theme)

    st.markdown("---")

    st.markdown(f"### 🔑 {t('Feature Analysis')}")
    col1, col2 = st.columns(2)
    with col1:
        feature_importance_chart(theme)
    with col2:
        nsf_distribution_chart(df, theme)

    st.markdown("---")

    st.markdown(f"### 🧠 {t('Model Explainability')}")
    st.markdown(t("Understand what drives the prediction with SHAP-based feature contributions."))
    shap_summary_chart(theme)

    st.markdown("---")

    st.markdown(f"### 📈 {t('Financial Relationships')}")
    col1, col2 = st.columns(2)
    with col1:
        financial_overview_chart(df, theme)
    with col2:
        model_performance_chart(theme)

    st.markdown("---")

    st.markdown(f"### 🔗 {t('Feature Correlation Heatmap')}")
    correlation_heatmap(df, theme)

    st.markdown("---")

    st.subheader(f"📋 {t('Dataset Preview')}")
    st.dataframe(df.head(10), use_container_width=True)


# =========================================================
# REPORTS PAGE
# =========================================================
elif page == "Reports":

    st.markdown(
        f'<div class="main-title">{t("Reports Center")}</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"## 📄 {t('Generated AI Reports')}")

    saved = st.session_state.get("saved_reports", [])

    if saved:
        for i, rep in enumerate(reversed(saved)):
            idx = len(saved) - 1 - i

            risk_clr = (
                "#10B981" if "LOW"    in rep["risk_label"]
                else "#F59E0B" if "MEDIUM" in rep["risk_label"]
                else "#EF4444"
            )
            box_bg  = "rgba(17,24,39,0.95)" if theme == "dark" else "#FFFFFF"
            box_bdr = "rgba(255,255,255,0.06)" if theme == "dark" else "#CBD5E1"

            col_info, col_btn = st.columns([3, 1])

            with col_info:
                st.markdown(
                    f"""
                    <div style="
                        background:{box_bg};
                        border-radius:14px;
                        padding:16px 20px;
                        border:1px solid {box_bdr};
                        border-left:4px solid {risk_clr};
                        margin-bottom:8px;
                    ">
                        <div style="font-weight:700;font-size:15px;color:{text};">
                            📊 {rep['filename']}
                        </div>
                        <div style="color:{badge_text};font-size:13px;margin-top:4px;">
                            Generated: {rep['timestamp']} ·
                            <span style="color:{risk_clr};font-weight:700;">
                                {rep['risk_label']}
                            </span> ·
                            Risk Score: <b>{rep['risk_score']}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_btn:
                st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label=t("📥 Download"),
                    data=rep["data"],
                    file_name=rep["filename"],
                    mime="application/pdf",
                    key=f"dl_report_{idx}"
                )
    else:
        st.info(
            t("No PDF reports generated yet. Run an analysis in the **Prediction** page to generate reports.")
        )

    st.markdown("---")

    st.markdown(f"## 📋 {t('Prediction History')}")

    history_df = load_history()

    if not history_df.empty:

        st.success(t("✅ {count} prediction(s) found in history").format(count=len(history_df)))

        h_cols = st.columns(3)
        h_data = [
            ("Total Predictions", str(len(history_df))),
            ("High Risk",         str((history_df["risk_label"] == "HIGH RISK").sum())),
            ("Avg Confidence",    f"{round(history_df['confidence'].mean(), 1)}%"),
        ]
        for col, (label, val) in zip(h_cols, h_data):
            with col:
                metric_card(label, val)

        st.markdown("")

        st.dataframe(
            history_df.sort_values("timestamp", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

        st.markdown("")

        act_col1, act_col2 = st.columns(2)

        with act_col1:
            csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                t("📥 Download Prediction History (CSV)"),
                csv,
                "prediction_history.csv",
                "text/csv"
            )

        with act_col2:
            if st.button(t("🗑️ Clear Prediction History")):
                history_df.iloc[0:0].to_csv("prediction_history.csv", index=False)
                st.success(t("✅ Prediction history cleared successfully"))
                st.rerun()

    else:
        st.info(t("No prediction history available yet."))
