import google.generativeai as genai
from dotenv import load_dotenv
import os

# NOTE: Do NOT raise at import time. Loading secrets during import
# causes the repo to fail when someone browses it on GitHub. We will
# load and validate the API key at call-time inside the generator so
# the app can still run without Gemini (recommendations will be
# skipped or a helpful message will be returned).


# =========================================================
# Helper: configure Gemini client lazily
# =========================================================
def _configure_genai():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return api_key
    except Exception:
        return None


# =========================================================
# AI RECOMMENDATION ENGINE
# =========================================================
def generate_ai_recommendations(
    result,
    input_df,
    rules_recommendations,
    lang="en"
):
    """Generate AI recommendations using Gemini.

    This function attempts to configure the Gemini client at runtime.
    If `GEMINI_API_KEY` is not set, the function returns a helpful
    message instead of raising an exception so that the rest of the
    app can work (predictions, analytics, etc.).
    """
    try:
        # If no data, bail early
        if input_df is None or input_df.empty:
            return "No input data provided for AI recommendations."

        # Lazy configure genai; if missing we return a helpful message
        key = _configure_genai()
        if not key:
            return (
                "Gemini API key not configured. To enable AI recommendations, "
                "set GEMINI_API_KEY in your environment or a local .env file."
            )

        row = input_df.iloc[0]

        rules_text = "\n".join([
            f"- {rec['title']}: {rec['description']}"
            for rec in rules_recommendations
        ])

        # Language instructions
        if lang == "ar":
            lang_instruction = (
                "You MUST respond entirely in Arabic (Modern Standard Arabic). "
                "All section headers, analysis, and recommendations must be in Arabic."
            )
            output_format = """
١. الملخص التنفيذي
٢. تحليل المخاطر
٣. تقييم التدفق النقدي
٤. توصية الإقراض
٥. استراتيجية التخفيف
٦. التحسينات التشغيلية
٧. القرار النهائي
"""
        else:
            lang_instruction = "Respond in English."
            output_format = """
1. Executive Summary
2. Risk Analysis
3. Cash Flow Assessment
4. Lending Recommendation
5. Mitigation Strategy
6. Operational Improvements
7. Final Decision
"""

        prompt = f"""
System: You are a senior SME banking risk consultant.
{lang_instruction}
Generate professional banking-grade recommendations based on the following data.

=====================================================
RISK ASSESSMENT
=====================================================
Risk Score: {result.get('risk_score')}
Risk Label: {result.get('risk_label')}
Confidence: {result.get('confidence')}

=====================================================
FINANCIAL DATA
=====================================================
{row.to_dict()}

=====================================================
RULES
=====================================================
{rules_text}

=====================================================
OUTPUT FORMAT
=====================================================
{output_format}
"""

        # Use a validated Gemini model
        model = genai.GenerativeModel(model_name="gemini-3.1-flash-lite")

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7
            )
        )

        return response.text

    except Exception as e:
        return f"AI recommendations failed: {str(e)}"