from flask import Flask, request, jsonify
import pandas as pd
import joblib
import random
import json

model = joblib.load('churn_model.pkl')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = pd.DataFrame(data, index=[0])

    soft_prediction = model.predict_proba(features)[:, 1]
    hard_prediction = model.predict(features)

    response = {
        'soft_prediction': soft_prediction.tolist(),
        'hard_prediction': int(hard_prediction[0])
    }

    return jsonify(response)


def generate_random_json():
    data = {
        'SeniorCitizen': random.randint(0, 1),
        'Partner': random.randint(0, 1),
        'Dependents': random.randint(0, 1),
        'PhoneService': random.randint(0, 1),
        'MultipleLines': random.randint(0, 1),
        'OnlineSecurity': random.randint(0, 1),
        'OnlineBackup': random.randint(0, 1),
        'DeviceProtection': random.randint(0, 1),
        'TechSupport': random.randint(0, 1),
        'StreamingTV': random.randint(0, 1),
        'StreamingMovies': random.randint(0, 1),
        'PaperlessBilling': random.randint(0, 1),
        'MonthlyCharges': round(random.uniform(20, 200), 2),
        'TotalCharges': round(random.uniform(20, 8000), 2),
        'HasInternetService': random.randint(0, 1),
        'HasContract': random.randint(0, 1),
        'Gender_Female': 0,
        'Gender_Male': 0,
        'InternetService_DSL': 0,
        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,
        'PaymentMethod_Bank transfer (automatic)': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0,
        'ContractType_Month-to-month': 0,
        'ContractType_One year': 0,
        'ContractType_Two year': 0,
        'TenureGroup_1 - 12': 0,
        'TenureGroup_13 - 24': 0,
        'TenureGroup_25 - 36': 0,
        'TenureGroup_37 - 48': 0,
        'TenureGroup_49 - 60': 0,
        'TenureGroup_61 - 72': 0
    }

    # Randomly assigning 1 to one of the gender options
    gender_key = random.choice(['Gender_Female', 'Gender_Male'])
    data[gender_key] = 1

    # Randomly assigning 1 to one of the internet service options
    internet_service_key = random.choice(['InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No'])
    data[internet_service_key] = 1

    # Randomly assigning 1 to one of the payment method options
    payment_method_key = random.choice(
        ['PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
         'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'])
    data[payment_method_key] = 1

    # Randomly assigning 1 to one of the contract type options
    contract_type_key = random.choice(['ContractType_Month-to-month', 'ContractType_One year', 'ContractType_Two year'])
    data[contract_type_key] = 1

    # Randomly assigning 1 to one of the tenure group options
    tenure_group_key = random.choice(
        ['TenureGroup_1 - 12', 'TenureGroup_13 - 24', 'TenureGroup_25 - 36', 'TenureGroup_37 - 48',
         'TenureGroup_49 - 60', 'TenureGroup_61 - 72'])
    data[tenure_group_key] = 1

    return json.dumps(data, indent=4)


@app.route('/rnd', methods=['GET'])
def rnd():
    return generate_random_json()


if __name__ == '__main__':
    app.run(debug=True)
