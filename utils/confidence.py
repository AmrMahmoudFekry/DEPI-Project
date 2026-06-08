def get_confidence_level(probability):

    if probability < 0.65:
        return 'Low'

    elif probability < 0.80:
        return 'Medium'

    return 'High'
