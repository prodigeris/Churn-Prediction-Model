from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
import random
import json

app = Flask(__name__)

model = joblib.load('logistic_model.pkl')

# OpenAPI definition

api = Api(app, version='1.0', title='Customer Churn Prediction API',
          description='A simple API for predicting customer churn')
customer_model = api.model('Customer', {
    'SeniorCitizen': fields.Integer(required=True, description='Senior Citizen binary indicator (0 or 1)'),
    'Partner': fields.Integer(required=True, description='Partner binary indicator (0 or 1)'),
    'Dependents': fields.Integer(required=True, description='Dependents binary indicator (0 or 1)'),
    'PhoneService': fields.Integer(required=True, description='Phone Service binary indicator (0 or 1)'),
    'MultipleLines': fields.Integer(required=True, description='Multiple Lines binary indicator (0 or 1)'),
    'OnlineSecurity': fields.Integer(required=True, description='Online Security binary indicator (0 or 1)'),
    'OnlineBackup': fields.Integer(required=True, description='Online Backup binary indicator (0 or 1)'),
    'DeviceProtection': fields.Integer(required=True, description='Device Protection binary indicator (0 or 1)'),
    'TechSupport': fields.Integer(required=True, description='Tech Support binary indicator (0 or 1)'),
    'StreamingTV': fields.Integer(required=True, description='Streaming TV binary indicator (0 or 1)'),
    'StreamingMovies': fields.Integer(required=True, description='Streaming Movies binary indicator (0 or 1)'),
    'PaperlessBilling': fields.Integer(required=True, description='Paperless Billing binary indicator (0 or 1)'),
    'MonthlyCharges': fields.Float(required=True, description='Monthly Charges (numeric value)'),
    'TotalCharges': fields.Float(required=True, description='Total Charges (numeric value)'),
    'HasInternetService': fields.Integer(required=True, description='Internet Service binary indicator (0 or 1)'),
    'HasContract': fields.Integer(required=True, description='Has Contract binary indicator (0 or 1)'),
    'Gender_Female': fields.Integer(required=True, description='Gender Female binary indicator (0 or 1)'),
    'Gender_Male': fields.Integer(required=True, description='Gender Male binary indicator (0 or 1)'),
    'InternetService_DSL': fields.Integer(required=True, description='Internet Service DSL binary indicator (0 or 1)'),
    'InternetService_Fiber optic': fields.Integer(required=True, description='Internet Service Fiber Optic binary indicator (0 or 1)'),
    'InternetService_No': fields.Integer(required=True, description='No Internet Service binary indicator (0 or 1)'),
    'PaymentMethod_Bank transfer (automatic)': fields.Integer(required=True, description='Payment Method Bank Transfer (automatic) binary indicator (0 or 1)'),
    'PaymentMethod_Credit card (automatic)': fields.Integer(required=True, description='Payment Method Credit Card (automatic) binary indicator (0 or 1)'),
    'PaymentMethod_Electronic check': fields.Integer(required=True, description='Payment Method Electronic Check binary indicator (0 or 1)'),
    'PaymentMethod_Mailed check': fields.Integer(required=True, description='Payment Method Mailed Check binary indicator (0 or 1)'),
    'ContractType_Month-to-month': fields.Integer(required=True, description='Contract Type Month-to-Month binary indicator (0 or 1)'),
    'ContractType_One year': fields.Integer(required=True, description='Contract Type One Year binary indicator (0 or 1)'),
    'ContractType_Two year': fields.Integer(required=True, description='Contract Type Two Year binary indicator (0 or 1)'),
    'TenureGroup_1 - 12': fields.Integer(required=True, description='Tenure Group 1-12 Months binary indicator (0 or 1)'),
    'TenureGroup_13 - 24': fields.Integer(required=True, description='Tenure Group 13-24 Months binary indicator (0 or 1)'),
    'TenureGroup_25 - 36': fields.Integer(required=True, description='Tenure Group 25-36 Months binary indicator (0 or 1)'),
    'TenureGroup_37 - 48': fields.Integer(required=True, description='Tenure Group 37-48 Months binary indicator (0 or 1)'),
    'TenureGroup_49 - 60': fields.Integer(required=True, description='Tenure Group 49-60 Months binary indicator (0 or 1)'),
    'TenureGroup_61 - 72': fields.Integer(required=True, description='Tenure Group 61-72 Months binary indicator (0 or 1)')
})

prediction_model = api.model('Prediction', {
    'soft_prediction': fields.List(fields.Float, description='Probability of churn'),
    'hard_prediction': fields.Integer(description='Binary prediction of churn')
})

ns = api.namespace('predictions', description='Churn Predictions')

# Prediction endpoint

@ns.route('/predict')
class Predict(Resource):
    @api.doc(responses={200: 'Success', 400: 'Validation Error'})
    @api.expect(customer_model)
    @api.marshal_with(prediction_model)
    def post(self):
        data = request.get_json()
        features = pd.DataFrame(data, index=[0])

        soft_prediction = model.predict_proba(features)[:, 1]
        hard_prediction = model.predict(features)

        return {
            'soft_prediction': soft_prediction.tolist(),
            'hard_prediction': int(hard_prediction[0])
        }

# Generates random data that the model could consume
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


# Random data endpoint
@app.route('/rnd', methods=['GET'])
def rnd():
    return generate_random_json()


if __name__ == '__main__':
    app.run(debug=True)
