from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import logging
import time
from model_training import HeartDiseasePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model at startup
predictor = HeartDiseasePredictor()

@app.before_first_request
def load_model():
    try:
        predictor.load_model('models')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        response = {
            'prediction': int(prediction[0]),
            'prediction_label': 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease',
            'probability': {
                'no_disease': float(probability[0][0]),
                'heart_disease': float(probability[0][1])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
