# =========================================================
# RULE-BASED RECOMMENDATION ENGINE  (bilingual)
# =========================================================
def generate_recommendations(input_df, result, lang="en"):

    row = input_df.iloc[0]
    recommendations = []

    # =====================================================
    # BILINGUAL CONTENT
    # =====================================================
    RECS = {
        "en": {
            "dti": {
                "title":       "Debt Optimization Strategy",
                "description": "The business currently operates with elevated leverage "
                               "exposure. Restructuring short-term liabilities and "
                               "improving liquidity allocation is strongly recommended.",
                "priority":    "HIGH"
            },
            "credit": {
                "title":       "Creditworthiness Enhancement",
                "description": "The owner's credit profile negatively impacts financing "
                               "eligibility. Improving repayment consistency and reducing "
                               "outstanding obligations may strengthen lending approval chances.",
                "priority":    "HIGH"
            },
            "volatility": {
                "title":       "Revenue Stabilization Plan",
                "description": "Revenue fluctuations indicate unstable operational cash flow. "
                               "Implement recurring revenue streams and reserve liquidity buffers.",
                "priority":    "MEDIUM"
            },
            "neg_days": {
                "title":       "Liquidity Management Improvement",
                "description": "Frequent negative balances suggest operational liquidity pressure. "
                               "Enhancing receivable collection efficiency is recommended.",
                "priority":    "HIGH"
            },
            "nsf": {
                "title":       "Operational Cash Buffer",
                "description": "NSF activity impacts financial reliability indicators. "
                               "Maintaining emergency liquidity reserves is advised.",
                "priority":    "MEDIUM"
            },
            "startup": {
                "title":       "Early-Stage Business Risk",
                "description": "Limited business operating history increases uncertainty. "
                               "Providing audited projections and operational evidence is recommended.",
                "priority":    "MEDIUM"
            },
            "high_risk": {
                "title":       "High-Risk Exposure",
                "description": "The overall financial profile indicates elevated lending risk. "
                               "A conservative financing approach is advised.",
                "priority":    "CRITICAL"
            },
            "healthy": {
                "title":       "Healthy Financial Position",
                "description": "The company demonstrates stable operational performance "
                               "and balanced financial indicators.",
                "priority":    "LOW"
            },
        },
        "ar": {
            "dti": {
                "title":       "استراتيجية تحسين الديون",
                "description": "تعمل الشركة حاليًا بمستوى مرتفع من الرافعة المالية. "
                               "يُوصى بشدة بإعادة هيكلة الالتزامات قصيرة الأجل "
                               "وتحسين توزيع السيولة.",
                "priority":    "HIGH"
            },
            "credit": {
                "title":       "تعزيز الجدارة الائتمانية",
                "description": "يؤثر الملف الائتماني للمالك سلبًا على أهلية التمويل. "
                               "قد يُسهم تحسين انتظام السداد وتقليل الالتزامات القائمة "
                               "في تعزيز فرص الحصول على القرض.",
                "priority":    "HIGH"
            },
            "volatility": {
                "title":       "خطة استقرار الإيرادات",
                "description": "تشير تذبذبات الإيرادات إلى تدفق نقدي تشغيلي غير مستقر. "
                               "يُنصح بتنويع مصادر الإيراد المتكررة وتكوين احتياطيات سيولة.",
                "priority":    "MEDIUM"
            },
            "neg_days": {
                "title":       "تحسين إدارة السيولة",
                "description": "تشير الأرصدة السلبية المتكررة إلى ضغط على السيولة التشغيلية. "
                               "يُوصى بتعزيز كفاءة تحصيل الذمم المدينة.",
                "priority":    "HIGH"
            },
            "nsf": {
                "title":       "احتياطي نقدي تشغيلي",
                "description": "يؤثر نشاط NSF على مؤشرات الموثوقية المالية. "
                               "يُنصح بالحفاظ على احتياطيات سيولة طارئة.",
                "priority":    "MEDIUM"
            },
            "startup": {
                "title":       "مخاطر الشركات الناشئة",
                "description": "يزيد محدودية السجل التشغيلي للشركة من حالة عدم اليقين. "
                               "يُوصى بتقديم توقعات مدققة وأدلة تشغيلية.",
                "priority":    "MEDIUM"
            },
            "high_risk": {
                "title":       "تعرض مرتفع للمخاطر",
                "description": "يشير الملف المالي الإجمالي إلى مخاطر إقراض مرتفعة. "
                               "يُنصح باتباع نهج تمويل محافظ.",
                "priority":    "CRITICAL"
            },
            "healthy": {
                "title":       "وضع مالي سليم",
                "description": "تُظهر الشركة أداءً تشغيليًا مستقرًا "
                               "ومؤشرات مالية متوازنة.",
                "priority":    "LOW"
            },
        }
    }

    rx = RECS.get(lang, RECS["en"])

    if row["dti_monthly"] > 0.45:
        recommendations.append(rx["dti"])

    if row["owner_credit_score"] < 600:
        recommendations.append(rx["credit"])

    if row["revenue_volatility_3m"] > 0.40:
        recommendations.append(rx["volatility"])

    if row["negative_days_3m"] > 15:
        recommendations.append(rx["neg_days"])

    if row["nsf_count_3m"] > 2:
        recommendations.append(rx["nsf"])

    if row["business_age_months"] < 24:
        recommendations.append(rx["startup"])

    if result["risk_score"] >= 70:
        recommendations.append(rx["high_risk"])

    if len(recommendations) == 0:
        recommendations.append(rx["healthy"])

    return recommendations