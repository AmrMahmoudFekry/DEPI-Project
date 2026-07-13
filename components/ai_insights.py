def generate_ai_insights(result, input_df, lang="en"):

    row = input_df.iloc[0]

    insights = []

    # =====================================================
    # BILINGUAL TEXT MAPS
    # =====================================================
    TEXTS = {
        "en": {
            "strong_income":    "Strong recurring income detected.",
            "reliable_owner":   "Owner credit profile is highly reliable.",
            "stable_revenue":   "Revenue stream appears stable.",
            "high_dti":         "Debt-to-income ratio is considered elevated.",
            "neg_balance":      "Frequent negative balance activity detected.",
            "nsf_risk":         "NSF activity may impact financial credibility.",
            "summary_low":      "AI assessment indicates a financially stable business profile.",
            "summary_medium":   "AI assessment indicates moderate financial risk.",
            "summary_high":     "AI assessment indicates elevated financial instability.",
        },
        "ar": {
            "strong_income":    "تم رصد دخل متكرر قوي.",
            "reliable_owner":   "الملف الائتماني للمالك موثوق للغاية.",
            "stable_revenue":   "يبدو تدفق الإيرادات مستقرًا.",
            "high_dti":         "نسبة الدين إلى الدخل تُعدّ مرتفعة.",
            "neg_balance":      "تم رصد نشاط متكرر برصيد سلبي.",
            "nsf_risk":         "نشاط NSF قد يؤثر على المصداقية المالية.",
            "summary_low":      "يشير تقييم الذكاء الاصطناعي إلى ملف تجاري مستقر ماليًا.",
            "summary_medium":   "يشير تقييم الذكاء الاصطناعي إلى مخاطر مالية متوسطة.",
            "summary_high":     "يشير تقييم الذكاء الاصطناعي إلى عدم استقرار مالي مرتفع.",
        }
    }

    tx = TEXTS.get(lang, TEXTS["en"])

    # =====================================================
    # POSITIVE SIGNALS
    # =====================================================
    if row["monthly_income_avg"] > 50000:
        insights.append(tx["strong_income"])

    if row["owner_credit_score"] > 700:
        insights.append(tx["reliable_owner"])

    if row["revenue_volatility_3m"] < 0.25:
        insights.append(tx["stable_revenue"])

    # =====================================================
    # RISK SIGNALS
    # =====================================================
    if row["dti_monthly"] > 0.45:
        insights.append(tx["high_dti"])

    if row["negative_days_3m"] > 10:
        insights.append(tx["neg_balance"])

    if row["nsf_count_3m"] > 2:
        insights.append(tx["nsf_risk"])

    # =====================================================
    # FINAL AI SUMMARY
    # =====================================================
    if result["risk_score"] < 40:
        summary = tx["summary_low"]
    elif result["risk_score"] < 70:
        summary = tx["summary_medium"]
    else:
        summary = tx["summary_high"]

    return summary, insights
