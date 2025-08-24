import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from model_training import HeartDiseasePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, model_dir='../models'):
        self.predictor = HeartDiseasePredictor()
        self.predictor.load_model('../models')
        self.explainer_lime = None
        
    def setup_lime_explainer(self, X_train):
        """Setup LIME explainer"""
        logger.info("Setting up LIME explainer...")
        X_train_scaled = self.predictor.scaler.transform(X_train)
        self.explainer_lime = LimeTabularExplainer(
            X_train_scaled,
            feature_names=self.predictor.feature_names,
            class_names=['No Disease', 'Heart Disease'],
            mode='classification'
        )
    
    def explain_with_shap(self, X_sample):
        """Generate SHAP explanations"""
        logger.info("Generating SHAP explanations...")
        X_sample_scaled = self.predictor.scaler.transform(X_sample)
        
        # Use LinearExplainer for logistic regression
        explainer = shap.LinearExplainer(self.predictor.model, X_sample_scaled)
        shap_values = explainer.shap_values(X_sample_scaled)
        
        return shap_values
    
    def explain_with_lime(self, X_sample, instance_idx=0):
        """Generate LIME explanations"""
        logger.info("Generating LIME explanations...")
        X_sample_scaled = self.predictor.scaler.transform(X_sample)
        
        def predict_fn(x):
            return self.predictor.model.predict_proba(x)
        
        explanation = self.explainer_lime.explain_instance(
            X_sample_scaled[instance_idx], 
            predict_fn, 
            num_features=len(self.predictor.feature_names)
        )
        
        return explanation
    
    def generate_plain_english_explanation(self, X_sample):
        """Generate plain English explanation of predictions"""
        logger.info("Generating plain English explanations...")
        
        explanations = []
        
        # Get SHAP values
        shap_values = self.explain_with_shap(X_sample)
        
        for i, sample in enumerate(X_sample.values):
            prediction, prob = self.predictor.predict(X_sample.iloc[[i]])
            
            explanation = f"Sample {i+1} Analysis:\n"
            explanation += f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}\n"
            explanation += f"Confidence: {prob[0][prediction[0]]:.2%}\n\n"
            
            explanation += "Key factors influencing this prediction:\n"
            
            # Get feature impacts
            sample_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values
            feature_impacts = list(zip(self.predictor.feature_names, sample_shap, sample))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for rank, (feature_name, shap_value, feature_value) in enumerate(feature_impacts[:5]):
                impact = "increases" if shap_value > 0 else "decreases"
                explanation += f"{rank+1}. {feature_name} = {feature_value:.1f} "
                explanation += f"({impact} risk by {abs(shap_value):.3f})\n"
            
            explanations.append(explanation)
        
        return explanations

def run_explainability_analysis():
    """Run complete explainability analysis"""
    import os
    os.chdir('..')  # Go to main directory
    
    # Load data
    predictor = HeartDiseasePredictor()
    X, y, df = predictor.load_and_preprocess_data('data/data.csv')
    
    # Get samples with heart disease for analysis
    disease_samples = df[df['target'] == 1].drop(['target', 'sno'], axis=1, errors='ignore')
    if 'gender' in disease_samples.columns and disease_samples['gender'].dtype == 'object':
        disease_samples['gender'] = pd.factorize(disease_samples['gender'])[0]
    
    disease_samples = disease_samples.head(5)  # First 5 samples
    
    # Setup explainer
    explainer = ModelExplainer()
    X_train = X.sample(n=100, random_state=42)  # Use subset for LIME setup
    explainer.setup_lime_explainer(X_train)
    
    # Generate explanations
    explanations = explainer.generate_plain_english_explanation(disease_samples)
    
    # Save explanations
    with open('heart_disease_explanations.txt', 'w') as f:
        f.write("HEART DISEASE PREDICTION EXPLANATIONS\n")
        f.write("="*50 + "\n\n")
        f.write("This analysis explains why the model predicts heart disease\n")
        f.write("for specific patient samples using SHAP values.\n\n")
        for explanation in explanations:
            f.write(explanation + "\n" + "-"*30 + "\n")
    
    logger.info("Explainability analysis completed. Results saved to heart_disease_explanations.txt")
    return True

if __name__ == "__main__":
    run_explainability_analysis()
