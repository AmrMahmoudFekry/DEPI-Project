import streamlit as st


def metric_card(title, value):

    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #111827, #1E293B);
        border-radius: 22px;
        padding: 28px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 10px 25px rgba(0,0,0,0.22);
        transition: 0.3s ease;
        margin-bottom: 8px;
    ">
        <div style="
            color: #CBD5E1;
            font-size: 17px;
            margin-bottom: 12px;
            font-family: Inter, sans-serif;
        ">
            {title}
        </div>
        <div style="
            color: white;
            font-size: 38px;
            font-weight: 800;
            font-family: Inter, sans-serif;
        ">
            {value}
        </div>
    </div>
    """

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )