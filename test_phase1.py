import sys
sys.path.append('src')

from model_training import HeartDiseasePredictor
import pandas as pd

def test_phase1():
    print("=" * 50)
    print("PHASE 1 TESTING")
    print("=" * 50)
    
    try:
        # Test model training
        print("1. Testing model training...")
        predictor = HeartDiseasePredictor()
        X, y, df = predictor.load_and_preprocess_data('data/data.csv')
        X_train, X_test, y_train, y_test, accuracy = predictor.train_model(X, y)
        predictor.save_model()
        print(f"✅ Model trained successfully with accuracy: {accuracy:.4f}")
        
        # Test model loading
        print("\n2. Testing model loading...")
        new_predictor = HeartDiseasePredictor()
        new_predictor.load_model()
        print("✅ Model loaded successfully")
        
        # Test prediction
        print("\n3. Testing predictions...")
        sample = X_test.iloc[[0]]
        pred, prob = new_predictor.predict(sample)
        print(f"✅ Prediction successful: {pred[0]} with probability {prob[0]}")
        
        print("\n" + "=" * 50)
        print("PHASE 1 COMPLETED SUCCESSFULLY! ✅")
        print("=" * 50)
        print("Next steps:")
        print("- Start Flask app: python src/simple_app.py")
        print("- Test API: curl http://localhost:8080/health")
        print("- Run Phase 2 when ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return False

if __name__ == "__main__":
    test_phase1()
