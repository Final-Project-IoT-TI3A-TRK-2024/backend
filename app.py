import json
import os
import time

import pymongo
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError
import paho.mqtt.client as mqtt

from utils import load_model

app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")  # Allow CORS for development
load_dotenv()

try:
    mongo = pymongo.MongoClient(os.getenv('MONGO_URI'))
    db = mongo['chart']
    collection = db['sensor']
except Exception as e:
    exit(f"Error: {str(e)}")


def publish_data(prediction_result):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(os.getenv('MQTT_USERNAME'), os.getenv('MQTT_PASSWORD'))
    client.connect(host=os.getenv('MQTT_HOST'), port=1883, keepalive=60)
    client.publish('iot/prediction', json.dumps({'prediction': bool(prediction_result)}))


class PredictSchema(Schema):
    crop_type = fields.Int(required=True)
    soil_moisture = fields.Float(required=True)
    temperature = fields.Float(required=True)
    humidity = fields.Float(required=True)


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()

    try:
        data = PredictSchema().load(json_data)
    except ValidationError as err:
        return jsonify({"errors": err.messages}), 422

    loaded_model = load_model("Decision Tree")
    prediction = loaded_model.predict([[data['crop_type'], data['soil_moisture'], data['temperature'], data['humidity']]])[0]
    publish_data(prediction)

    return jsonify({
        'prediction': bool(prediction)
    }), 200


def send_real_time_data():
    while True:
        data = collection.find({}).sort('_id', pymongo.DESCENDING).to_list()
        for d in data:
            d.pop('_id')
        socketio.emit('data', {'data': json.dumps(data)})
        time.sleep(5)


@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.start_background_task(send_real_time_data)


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    app.run(debug=True, host='0.0.0.0')
