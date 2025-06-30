from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from pymongo import MongoClient

app = Flask(__name__)

model = load_model("model/admission_model.h5")
scaler = joblib.load("model/scaler.pkl")

client = MongoClient("mongodb://localhost:27017/")
db = client['admissionDB']
collection = db['predictions']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        print("JSON Payload:", data)

        features = [
            float(data['GRE Score']),
            float(data['TOEFL Score']),
            float(data['University Rating']),
            float(data['SOP']),
            float(data['LOR ']),  # or fix to 'LOR' everywhere
            float(data['CGPA']),
            float(data['Research'])
        ]

        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0][0]

        # Convert NumPy float32 → Python float for JSON + MongoDB
        prediction = float(prediction)
        data["prediction"] = prediction
        collection.insert_one(data)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400





@app.route('/records', methods=['GET'])
def get_records():
    records = list(collection.find({}, {"_id": 0}))
    return jsonify(records)

if __name__ == "__main__":
    app.run(debug=True)
