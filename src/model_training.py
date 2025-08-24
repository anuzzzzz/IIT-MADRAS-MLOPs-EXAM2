import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the heart disease dataset"""
        logger.info("Loading data from: %s", data_path)
        df = pd.read_csv(data_path)
        
        print("Original columns:", df.columns.tolist())
        print("Data shape:", df.shape)
        print("Target values:", df['target'].unique())
        
        # Handle gender encoding
        if df['gender'].dtype == 'object':
            df['gender'] = pd.factorize(df['gender'])[0]
        
        # Check for missing values
        print("Missing values per column:")
        print(df.isnull().sum())
        
        # Drop rows with missing values
        cleaned_df = df.dropna()
        logger.info("Dataset shape after cleaning: %s", cleaned_df.shape)
        
        # Separate features and target
        X = cleaned_df.drop(['target', 'sno'], axis=1, errors='ignore')  # Remove sno if exists
        y = cleaned_df['target']
        
        # Convert target to binary if needed
        if y.dtype == 'object':
            y = (y == 'yes').astype(int)
        
        self.feature_names = X.columns.tolist()
        print("Features:", self.feature_names)
        
        return X, y, cleaned_df
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train the logistic regression model"""
        logger.info("Starting model training...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Simple model first (we'll add hyperparameter tuning later)
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info("Test accuracy: %.4f", accuracy)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train, X_test, y_train, y_test, accuracy
    
    def save_model(self, model_dir='models'):
        """Save trained model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'heart_disease_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(self.feature_names, os.path.join(model_dir, 'feature_names.pkl'))
        logger.info("Model saved to %s", model_dir)
    
    def load_model(self, model_dir='models'):
        """Load trained model and scaler"""
        self.model = joblib.load(os.path.join(model_dir, 'heart_disease_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        logger.info("Model loaded from %s", model_dir)
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

# Test the model training
if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    
    # Load and train
    X, y, df = predictor.load_and_preprocess_data('data/data.csv')
    X_train, X_test, y_train, y_test, accuracy = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    # Test prediction on first sample
    sample = X_test.iloc[[0]]
    pred, prob = predictor.predict(sample)
    print(f"\nTest prediction: {pred[0]} (probability: {prob[0]})")
    
    print(f"\nPhase 1 Complete! Model accuracy: {accuracy:.4f}")
