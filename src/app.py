from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import logging
import time
import os
import sys
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from model_training import HeartDiseasePredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('heart_disease_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('heart_disease_prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('heart_disease_errors_total', 'Total errors encountered')

app = Flask(__name__)

# Global model instance
predictor = None

def load_model():
    """Load the trained model"""
    global predictor
    try:
        predictor = HeartDiseasePredictor()
        model_path = '/app/models' if os.path.exists('/app/models') else 'models'
        predictor.load_model(model_path)
        logger.info("Model loaded successfully from %s", model_path)
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

# Load model at startup
load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = predictor is not None and predictor.model is not None
    return jsonify({
        'status': 'healthy' if model_status else 'unhealthy',
        'timestamp': time.time(),
        'model_loaded': model_status
    }), 200 if model_status else 503

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with logging and metrics"""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if predictor is None or predictor.model is None:
            ERROR_COUNTER.inc()
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            ERROR_COUNTER.inc()
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            ERROR_COUNTER.inc()
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Convert gender if string
        if isinstance(data.get('gender'), str):
            data['gender'] = 0 if data['gender'].lower() == 'male' else 1
        
        # Create DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        # Log prediction with sample details
        logger.info(f"Prediction made: {prediction[0]} (confidence: {probability[0][prediction[0]]:.3f}) "
                   f"for patient age={data['age']}, gender={data['gender']}")
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        # Return response
        response = {
            'prediction': int(prediction[0]),
            'prediction_label': 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease',
            'probability': {
                'no_disease': float(probability[0][0]),
                'heart_disease': float(probability[0][1])
            },
            'confidence': float(max(probability[0])),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'samples' not in data:
            ERROR_COUNTER.inc()
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        results = []
        
        for i, sample in enumerate(samples):
            try:
                # Convert gender if needed
                if isinstance(sample.get('gender'), str):
                    sample['gender'] = 0 if sample['gender'].lower() == 'male' else 1
                
                input_data = pd.DataFrame([sample])
                prediction, probability = predictor.predict(input_data)
                
                result = {
                    'sample_id': i,
                    'prediction': int(prediction[0]),
                    'prediction_label': 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease',
                    'probability': {
                        'no_disease': float(probability[0][0]),
                        'heart_disease': float(probability[0][1])
                    },
                    'confidence': float(max(probability[0]))
                }
                
                results.append(result)
                PREDICTION_COUNTER.inc()
                
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'error': str(e)
                })
        
        logger.info(f"Batch prediction completed: {len(results)} samples processed")
        
        return jsonify({
            'results': results,
            'total_samples': len(samples),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'timestamp': time.time()
        })
        
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if predictor is None or predictor.model is None:
            return jsonify({'error': 'Model not loaded'}), 503
            
        return jsonify({
            'model_type': 'Logistic Regression',
            'features': predictor.feature_names,
            'feature_count': len(predictor.feature_names),
            'target_classes': ['No Heart Disease', 'Heart Disease'],
            'model_loaded': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Production-ready settings
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
