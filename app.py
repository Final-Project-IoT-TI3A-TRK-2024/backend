from flask import Flask, request, jsonify
from utils import load_model
from marshmallow import Schema, fields, ValidationError

app = Flask(__name__)


class PredictSchema(Schema):
    model = fields.Str(required=True)
    N = fields.Float(required=True)
    P = fields.Float(required=True)
    K = fields.Float(required=True)
    temperature = fields.Float(required=True)
    humidity = fields.Float(required=True)
    ph = fields.Float(required=True)
    rainfall = fields.Float(required=True)


@app.route('/predict', methods=['POST'])
def predict():
    models = {
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'logistic_regression': 'Logistic Regression',
        'support_vector_machine': 'SVM',
        'xgboost': 'XGBoost',
        'naive_bayes': 'Naive Bayes',
    }

    json_data = request.get_json()

    try:
        data = PredictSchema().load(json_data)
    except ValidationError as err:
        return jsonify({"errors": err.messages}), 422

    if data['model'] in models:
        loaded_model = load_model(models[data['model']])
        prediction = loaded_model.predict([[data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']]])[0]
        return jsonify({
            'model': models[data['model']],
            'prediction': prediction
        }), 200
    else:
        return jsonify({"error": "Invalid model name"}), 422


if __name__ == '__main__':
    app.run(debug=True)
