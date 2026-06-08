import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
import json
from pathlib import Path

from utils.model_loader import load_pipeline


# =========================================================
# COMMON LAYOUT — accepts optional theme param
# =========================================================
def apply_layout(fig, theme="dark", height=420):

    if theme == "light":
        bg        = "#FFFFFF"
        plot_bg   = "#F8FAFC"
        font_clr  = "#0F172A"
        title_clr = "#020617"
        tick_clr  = "#334155"
    else:
        bg        = "rgba(0,0,0,0)"
        plot_bg   = "rgba(0,0,0,0)"
        font_clr  = "#F8FAFC"
        title_clr = "#F1F5F9"
        tick_clr  = "#CBD5E1"

    fig.update_layout(
        template="plotly_white" if theme == "light" else "plotly_dark",
        paper_bgcolor=bg,
        plot_bgcolor=plot_bg,
        height=height,
        font=dict(color=font_clr, size=13),
        title_font=dict(size=18, color=title_clr),
        xaxis=dict(
            tickfont=dict(color=tick_clr),
            title_font=dict(color=tick_clr),
            gridcolor="rgba(100,100,100,0.15)"
        ),
        yaxis=dict(
            tickfont=dict(color=tick_clr),
            title_font=dict(color=tick_clr),
            gridcolor="rgba(100,100,100,0.15)"
        ),
        legend=dict(font=dict(color=font_clr)),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# =========================================================
# 1. RISK DISTRIBUTION — Grouped Bar (clean & readable)
# =========================================================
def risk_distribution(df, theme="dark"):

    counts = df["risk_sharp"].value_counts().reset_index()
    counts.columns = ["Risk Class", "Count"]
    counts["Label"] = counts["Risk Class"].map({0: "Low Risk (0)", 1: "High Risk (1)"})
    counts["Color"] = counts["Risk Class"].map({0: "#10B981", 1: "#EF4444"})

    fig = go.Figure()

    for _, row in counts.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Label"]],
            y=[row["Count"]],
            name=row["Label"],
            marker_color=row["Color"],
            text=[f"{row['Count']:,}"],
            textposition="outside",
            textfont=dict(size=14, color="#F8FAFC" if theme == "dark" else "#020617")
        ))

    fig.update_layout(
        title="Class Distribution (Target Variable)",
        showlegend=False,
        bargap=0.35
    )

    fig = apply_layout(fig, theme, height=380)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================
# 2. RISK SEGMENTS PIE
# =========================================================
def risk_segments_chart(df, theme="dark"):

    label_map = {0: "Low Risk", 1: "High Risk"}
    df2 = df.copy()
    df2["Risk Label"] = df2["risk_sharp"].map(label_map)

    fig = px.pie(
        df2,
        names="Risk Label",
        title="Risk Segment Share",
        hole=0.50,
        color="Risk Label",
        color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"}
    )

    fig.update_traces(
        textinfo="percent+label",
        textfont=dict(
            size=13,
            color="#0F172A" if theme == "light" else "#F8FAFC"
        ),
        pull=[0.04, 0.04]
    )

    fig = apply_layout(fig, theme, height=380)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================
# 3. FEATURE IMPORTANCE (from model knowledge)
# =========================================================
def feature_importance_chart(theme="dark"):
    """
    Shows the top feature importances from the trained model
    (hardcoded from the Modeling.ipynb output — great for presentations).
    """
    features = [
        "nsf_count_3m",
        "negative_days_3m",
        "negative_activity_ratio",
        "dti_monthly",
        "monthly_income_avg",
        "revenue_volatility_3m",
        "total_deposits_3m",
        "request_ratio",
        "stability_score",
        "owner_credit_score",
    ]
    importance = [
        0.289, 0.113, 0.107, 0.073, 0.068,
        0.063, 0.062, 0.045, 0.039, 0.036,
    ]

    colors_list = ["#2563EB" if v > 0.08 else "#06B6D4" for v in importance]

    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation="h",
        marker_color=colors_list,
        text=[f"{v:.1%}" for v in importance],
        textposition="outside",
        textfont=dict(color="#F8FAFC" if theme == "dark" else "#020617", size=11)
    ))

    fig.update_layout(
        title="Top 10 Feature Importances (XGBoost)",
        xaxis_title="Importance Score",
        yaxis=dict(autorange="reversed")
    )

    fig = apply_layout(fig, theme, height=420)
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 4. DTI vs CREDIT AMOUNT — scatter by risk
# =========================================================
def financial_overview_chart(df, theme="dark"):

    sample = df.sample(min(1500, len(df)), random_state=42)
    label_map = {0: "Low Risk", 1: "High Risk"}
    sample = sample.copy()
    sample["Risk Label"] = sample["risk_sharp"].map(label_map)

    fig = px.scatter(
        sample,
        x="dti_monthly",
        y="monthly_income_avg",
        color="Risk Label",
        title="Debt-to-Income vs Monthly Income by Risk",
        opacity=0.65,
        color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"},
        labels={
            "dti_monthly": "Debt-to-Income Ratio",
            "monthly_income_avg": "Monthly Income (EGP)"
        }
    )

    fig = apply_layout(fig, theme, height=400)
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 5. NSF COUNT DISTRIBUTION BY RISK
# =========================================================
def nsf_distribution_chart(df, theme="dark"):

    label_map = {0: "Low Risk", 1: "High Risk"}
    df2 = df.copy()
    df2["Risk Label"] = df2["risk_sharp"].map(label_map)

    fig = px.histogram(
        df2,
        x="nsf_count_3m",
        color="Risk Label",
        barmode="overlay",
        opacity=0.75,
        title="NSF Count Distribution by Risk Class",
        color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"},
        labels={"nsf_count_3m": "NSF Count (3 Months)"}
    )

    fig = apply_layout(fig, theme, height=380)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =========================================================
# 6. CORRELATION HEATMAP (key features only)
# =========================================================
def correlation_heatmap(df, theme="dark"):

    key_cols = [
        "credit_amount", "monthly_income_avg",
        "total_deposits_3m", "dti_monthly",
        "nsf_count_3m", "negative_days_3m",
        "revenue_volatility_3m", "owner_credit_score",
        "risk_sharp"
    ]

    cols = [c for c in key_cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)

    # Rename risk_sharp for readability
    corr = corr.rename(
        index={"risk_sharp": "RISK"},
        columns={"risk_sharp": "RISK"}
    )

    text_clr = "#0F172A" if theme == "light" else "#F8FAFC"

    heatmap_scale = (
        "RdBu" if theme == "light" else [
            [0.0, "#8B0000"],
            [0.35, "#F87171"],
            [0.5, "#0F172A"],
            [0.65, "#60A5FA"],
            [1.0, "#1D4ED8"]
        ]
    )

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=heatmap_scale,
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap"
    )

    fig.update_traces(textfont=dict(size=10, color=text_clr), hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>")

    fig = apply_layout(fig, theme, height=480)

    fig.update_coloraxes(colorbar=dict(
        bgcolor="rgba(15,23,42,0.8)",
        tickfont=dict(color="#0F172A" if theme == "light" else "#F8FAFC"),
        title=dict(font=dict(color="#0F172A" if theme == "light" else "#F8FAFC")),
        outlinecolor="#475569" if theme == "light" else "#CBD5E1"
    ))

    st.plotly_chart(fig, use_container_width=True)


def shap_summary_chart(theme="dark"):
    root_path = Path(__file__).resolve().parent.parent
    data_path = root_path / 'Data' / 'SMEs_Data.csv'

    if not data_path.exists():
        st.info("SHAP feature chart unavailable because the dataset file is missing.")
        return

    try:
        df = pd.read_csv(data_path)
        sample = df.drop(columns=['risk_sharp'], errors='ignore').sample(
            n=min(200, len(df)), random_state=42
        )

        pipeline = load_pipeline()
        prepared = pipeline.named_steps['feat_eng'].transform(sample)
        feature_names = list(prepared.columns)
        processed = pipeline.named_steps['preprocessor'].transform(prepared)

        classifier = pipeline.named_steps['classifier']
        if hasattr(classifier, 'calibrated_classifiers_'):
            estimator = classifier.calibrated_classifiers_[0].estimator
        else:
            estimator = getattr(classifier, 'base_estimator', classifier)

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_shap = shap_values.mean(axis=0)
        importance = np.abs(mean_shap)
        direction = np.sign(mean_shap)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'direction': direction
        }).sort_values('importance', ascending=False).head(12)

        importance_df['color'] = importance_df['direction'].map({
            1.0: '#EF4444',
            -1.0: '#10B981',
            0.0: '#6B7280'
        })

        st.markdown(
            "<div style='padding:0 12px; margin-bottom:12px; color: #CBD5E1;'>"
            "هذا الرسم يوضح أقوى متغيرات تؤثر على قرار المخاطر. القيم الحمراء تدفع نحو خطر أعلى، والقيم الخضراء تدعم انخفاض الخطر."
            "</div>",
            unsafe_allow_html=True
        )

        fig = go.Figure(go.Bar(
            x=importance_df['importance'][::-1],
            y=importance_df['feature'][::-1],
            orientation='h',
            marker_color=importance_df['color'][::-1],
            text=[f"{v:.3f}" for v in importance_df['importance'][::-1]],
            textposition='outside'
        ))
        fig.update_layout(
            title='SHAP Feature Contribution (Top 12)',
            xaxis_title='Mean |SHAP value|',
            yaxis_title='',
        )
        fig = apply_layout(fig, theme, height=420)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Unable to generate SHAP explanation: {e}")


# =========================================================
# 7. MODEL COMPARISON CHART (dynamic)

def model_comparison_chart(theme="dark"):
    """Render the model comparison chart based on model_comparison.csv."""

    root_path = Path(__file__).resolve().parent.parent
    comparison_path = root_path / 'model_comparison.csv'

    if not comparison_path.exists():
        st.info(
            "Run `modeling_pipeline.py` first to generate `model_comparison.csv` and see model comparison here."
        )
        return

    df = pd.read_csv(comparison_path)
    if df.empty:
        st.info("Model comparison file is present but empty.")
        return

    df = df.sort_values('test_roc_auc', ascending=False)
    top_model = df.iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['model'],
        y=df['test_roc_auc'],
        name='Test ROC AUC',
        marker_color='#2563EB',
        text=[f"{v:.2f}%" for v in df['test_roc_auc']],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=df['model'],
        y=df['cv_roc_auc_mean'],
        name='CV ROC AUC',
        marker_color='#06B6D4',
        text=[f"{v:.2f}%" for v in df['cv_roc_auc_mean']],
        textposition='outside'
    ))

    fig.update_layout(
        title='Model Comparison — Test vs CV ROC AUC',
        barmode='group',
        yaxis=dict(title='ROC AUC (%)', range=[max(80, min(df['cv_roc_auc_mean'].min(), df['test_roc_auc'].min()) - 2), 101]),
        xaxis_title='',
        legend_title_text='Metric'
    )
    fig = apply_layout(fig, theme, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Best Model:** {top_model['model']} · "
        f"CV ROC AUC: {top_model['cv_roc_auc_mean']:.2f}% · "
        f"Test ROC AUC: {top_model['test_roc_auc']:.2f}%"
    )


# =========================================================
# 8. MODEL PERFORMANCE SUMMARY (static, for presentation)
# =========================================================
def model_performance_chart(theme="dark"):
    """Bar chart showing model metrics from the latest training output."""

    root_path = Path(__file__).resolve().parent.parent
    metrics_path = root_path / "model_comparison_results.json"

    default_values = [98.40, 98.08, 98.50, 98.29, 99.92]
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    values = default_values

    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            final_metrics = data.get("final_metrics", {})
            values = [
                final_metrics.get("test_accuracy", default_values[0]),
                final_metrics.get("test_precision", default_values[1]),
                final_metrics.get("test_recall", default_values[2]),
                final_metrics.get("test_f1", default_values[3]),
                final_metrics.get("test_roc_auc", default_values[4]),
            ]
        except Exception:
            values = default_values

    text_color = "#F8FAFC" if theme == "dark" else "#020617"

    fig = go.Figure(go.Bar(
        x=metrics,
        y=values,
        marker_color=[
            "#10B981", "#2563EB", "#06B6D4", "#8B5CF6", "#F59E0B"
        ],
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=13, color=text_color)
    ))

    min_val = min(values) if values else 0
    max_val = max(values) if values else 100
    y_lower = max(0, min_val - 3)
    y_upper = min(105, max_val + 4)

    fig.update_layout(
        title="Model Performance Metrics (Test Set)",
        yaxis=dict(range=[y_lower, y_upper], title="Score (%)"),
        xaxis_title="Metric",
        margin=dict(l=40, r=40, t=80, b=80)
    )

    fig = apply_layout(fig, theme, height=380)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})