def build_risk_explanation(row, probability, risk_label, confidence, color):
    explanations = []

    # High rejected transactions
    if row.get('rejected_transactions_3m', 0) > 10:
        explanations.append(
            'High rejected transaction activity detected.'
        )

    # Negative balance periods
    if row.get('negative_days_3m', 0) > 15:
        explanations.append(
            'Business shows recurring negative balance periods.'
        )

    # Debt-to-income ratio
    if row.get('dti_monthly', 0) > 0.50:
        explanations.append(
            'Debt-to-income ratio exceeds preferred threshold.'
        )

    # Credit score
    if row.get('owner_credit_score', 1000) < 600:
        explanations.append(
            'Owner credit profile is considered weak.'
        )

    # Revenue volatility
    if row.get('revenue_volatility_3m', 0) > 0.60:
        explanations.append(
            'Revenue stream appears unstable.'
        )

    # Default case
    if len(explanations) == 0:
        explanations.append(
            'Financial indicators appear stable.'
        )

    explanation = ' '.join(explanations)

    return {
        'risk_score': round(probability * 100, 2),
        'risk_label': risk_label,
        'confidence': confidence,
        'explanation': explanation,
        'color': color
    }