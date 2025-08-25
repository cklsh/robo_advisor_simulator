def assign_risk_profile(row):
    """ this function is to assign the risk profile (Conservative, Moderate, Aggresive)
    based on the scoring of demographic and financial feature"""
    score = 0

    #Age scoring
    if row['age'] <= 30:
        score += 2
    elif row['age'] <= 50:
        score += 1

    #Job scoring
    high_risk_job=['management', 'entrepreneur', 'self-employed']
    medium_risk_job=['technician', 'services', 'admin']

    if row['job'] in high_risk_job:
        score += 2
    elif row['job'] in medium_risk_job:
        score += 1

    #Loan-based deduction
    if row['housing'] == 'yes': #Interpreted as has a housing loan
        score -= 1
    elif row['loan'] == 'yes':
        score -= 1
    
    #Education Level
    if row['education'] == 'tertiary':
        score += 1

    #Risk Assignment
    if score >= 4:
        return 'Aggresive'
    elif score >= 2:
        return 'Moderate'
    else:
        return 'Conservative'

    