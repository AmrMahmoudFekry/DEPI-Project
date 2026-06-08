import streamlit as st


def risk_badge(result, lang="en"):
    """Render the coloured risk badge.

    Uses result['risk_label'] directly — which is already translated
    by model_loader.predict_risk(lang=lang). No extra mapping needed.
    """
    color = result["color"]
    label = result["risk_label"]

    bg = {
        "green":  "#10B981",
        "orange": "#F59E0B",
        "red":    "#EF4444"
    }

    st.markdown(
        f"""
        <div style="
            background:{bg[color]};
            padding:18px;
            border-radius:14px;
            text-align:center;
            color:white;
            font-size:22px;
            font-weight:bold;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )