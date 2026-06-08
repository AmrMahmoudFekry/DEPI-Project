import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# =========================================================
# GAUGE CHART
# =========================================================
def create_gauge_chart(risk_score, theme="dark"):

    bg_color = "rgba(0,0,0,0)" if theme == "dark" else "#FFFFFF"
    font_color = "#F8FAFC" if theme == "dark" else "#0F172A"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            title={
                "text": "Risk Probability",
                "font": {"color": font_color, "size": 16}
            },
            number={"suffix": "%", "font": {"color": font_color, "size": 32}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": font_color,
                    "tickfont": {"color": font_color}
                },
                "bar": {"color": "#2563EB", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],  "color": "rgba(16,185,129,0.25)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.25)"},
                    {"range": [70, 100],"color": "rgba(239,68,68,0.25)"}
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.75,
                    "value": risk_score
                }
            }
        )
    )

    fig.update_layout(
        height=320,
        paper_bgcolor=bg_color,
        font=dict(color=font_color),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# =========================================================
# RISK BREAKDOWN BAR CHART
# =========================================================
def create_risk_breakdown_chart(input_df, theme="dark"):

    row = input_df.iloc[0]

    categories = [
        "Debt Risk",
        "NSF Activity",
        "Liquidity Gap",
        "Credit Weakness",
        "Revenue Instability"
    ]

    values = [
        min(row["dti_monthly"] * 100, 100),
        min(row["nsf_count_3m"] * 12, 100),
        min(row["negative_days_3m"] * 3.3, 100),
        min(100 - (row["owner_credit_score"] / 8.5), 100),
        min(row["revenue_volatility_3m"] * 100, 100)
    ]

    colors_list = ["#EF4444" if v > 60 else "#F59E0B" if v > 30 else "#10B981"
                   for v in values]

    font_color = "#F8FAFC" if theme == "dark" else "#0F172A"
    bg_color   = "rgba(0,0,0,0)" if theme == "dark" else "#FFFFFF"

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors_list,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=font_color, size=12)
    ))

    fig.update_layout(
        title="Risk Factor Analysis",
        yaxis=dict(
            range=[0, 115],
            title="Risk Score",
            gridcolor="rgba(100,100,100,0.15)",
            tickfont=dict(color=font_color),
            title_font=dict(color=font_color)
        ),
        xaxis=dict(
            tickfont=dict(color=font_color),
            title_font=dict(color=font_color)
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        title_font=dict(color=font_color, size=16),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white" if theme == "light" else "plotly_dark"
    )

    return fig


# =========================================================
# FINANCIAL TREND CHART (uses real input data context)
# =========================================================
def create_trend_chart(input_df=None, theme="dark"):
    """
    Shows a simulated 6-month trend based on the input profile.
    If no input_df, falls back to random data.
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    font_color = "#F8FAFC" if theme == "dark" else "#0F172A"
    bg_color   = "rgba(0,0,0,0)" if theme == "dark" else "#FFFFFF"

    if input_df is not None:
        row = input_df.iloc[0]
        base = row.get("monthly_income_avg", 50000)
        vol  = row.get("revenue_volatility_3m", 0.3)
        np.random.seed(42)
        income_trend = [
            round(base * (1 + np.random.uniform(-vol, vol * 0.5)))
            for _ in months
        ]
        deposit_trend = [
            round(row.get("total_deposits_3m", 100000) / 3 * (1 + np.random.uniform(-0.1, 0.15)))
            for _ in months
        ]
    else:
        np.random.seed(42)
        income_trend  = np.random.randint(40000, 90000, size=6).tolist()
        deposit_trend = np.random.randint(30000, 80000, size=6).tolist()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months, y=income_trend,
        mode="lines+markers",
        name="Monthly Income",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=7)
    ))

    fig.add_trace(go.Scatter(
        x=months, y=deposit_trend,
        mode="lines+markers",
        name="Monthly Deposits",
        line=dict(color="#10B981", width=2.5, dash="dot"),
        marker=dict(size=7)
    ))

    fig.update_layout(
        title="6-Month Financial Trend (Simulated)",
        yaxis=dict(
            title="Amount (EGP)",
            tickfont=dict(color=font_color),
            title_font=dict(color=font_color),
            gridcolor="rgba(100,100,100,0.15)"
        ),
        xaxis=dict(
            tickfont=dict(color=font_color),
            title_font=dict(color=font_color)
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        title_font=dict(color=font_color, size=16),
        legend=dict(font=dict(color=font_color)),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white" if theme == "light" else "plotly_dark"
    )

    return fig