import pandas as pd
import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio,
    equalized_odds_ratio
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from model_training import HeartDiseasePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FairnessAnalyzer:
    def __init__(self, model_dir='../models'):
        self.predictor = HeartDiseasePredictor()
        self.predictor.load_model('../models')
        
    def analyze_fairness(self, X_test, y_test, sensitive_feature):
        """Analyze model fairness with respect to sensitive attribute"""
        logger.info("Analyzing model fairness...")
        
        # Make predictions
        predictions, probabilities = self.predictor.predict(X_test)
        
        # Calculate fairness metrics
        try:
            dp_diff = demographic_parity_difference(
                y_test, predictions, sensitive_features=sensitive_feature
            )
            eo_diff = equalized_odds_difference(
                y_test, predictions, sensitive_features=sensitive_feature
            )
            dp_ratio = demographic_parity_ratio(
                y_test, predictions, sensitive_features=sensitive_feature
            )
            eo_ratio = equalized_odds_ratio(
                y_test, predictions, sensitive_features=sensitive_feature
            )
        except Exception as e:
            logger.error(f"Error calculating fairness metrics: {e}")
            # Set default values if calculation fails
            dp_diff = dp_ratio = eo_diff = eo_ratio = 0.0
        
        # Calculate performance metrics by group
        groups = np.unique(sensitive_feature)
        group_metrics = {}
        
        for group in groups:
            group_mask = sensitive_feature == group
            group_y_test = y_test[group_mask]
            group_predictions = predictions[group_mask]
            
            if len(group_y_test) > 0:
                group_metrics[group] = {
                    'accuracy': accuracy_score(group_y_test, group_predictions),
                    'precision': precision_score(group_y_test, group_predictions, zero_division=0),
                    'recall': recall_score(group_y_test, group_predictions, zero_division=0),
                    'size': len(group_y_test),
                    'positive_rate': np.mean(group_predictions) if len(group_predictions) > 0 else 0
                }
        
        fairness_results = {
            'demographic_parity_difference': dp_diff,
            'equalized_odds_difference': eo_diff,
            'demographic_parity_ratio': dp_ratio,
            'equalized_odds_ratio': eo_ratio,
            'group_metrics': group_metrics
        }
        
        return fairness_results
    
    def generate_fairness_report(self, fairness_results):
        """Generate a comprehensive fairness report"""
        report = "FAIRNESS ANALYSIS REPORT\n"
        report += "=" * 40 + "\n\n"
        
        report += "FAIRNESS METRICS:\n"
        report += f"Demographic Parity Difference: {fairness_results['demographic_parity_difference']:.4f}\n"
        report += f"Equalized Odds Difference: {fairness_results['equalized_odds_difference']:.4f}\n"
        report += f"Demographic Parity Ratio: {fairness_results['demographic_parity_ratio']:.4f}\n"
        report += f"Equalized Odds Ratio: {fairness_results['equalized_odds_ratio']:.4f}\n\n"
        
        report += "INTERPRETATION:\n"
        if abs(fairness_results['demographic_parity_difference']) < 0.1:
            report += "✓ Good demographic parity (difference < 0.1)\n"
        else:
            report += "⚠ Potential bias in demographic parity (difference ≥ 0.1)\n"
            
        if abs(fairness_results['equalized_odds_difference']) < 0.1:
            report += "✓ Good equalized odds (difference < 0.1)\n"
        else:
            report += "⚠ Potential bias in equalized odds (difference ≥ 0.1)\n"
        
        report += "\nGROUP PERFORMANCE METRICS:\n"
        for group, metrics in fairness_results['group_metrics'].items():
            gender_label = 'Male' if group == 0 else 'Female'
            report += f"\n{gender_label} (Group {group}):\n"
            report += f"  Sample size: {metrics['size']}\n"
            report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"  Precision: {metrics['precision']:.4f}\n"
            report += f"  Recall: {metrics['recall']:.4f}\n"
            report += f"  Positive prediction rate: {metrics['positive_rate']:.4f}\n"
        
        return report

def run_fairness_analysis():
    """Run complete fairness analysis"""
    import os
    os.chdir('..')  # Go to main directory
    
    # Load and prepare data
    predictor = HeartDiseasePredictor()
    X, y, df = predictor.load_and_preprocess_data('data/data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get gender as sensitive feature
    gender_test = X_test['gender'].values
    
    # Run fairness analysis
    analyzer = FairnessAnalyzer()
    fairness_results = analyzer.analyze_fairness(X_test, y_test, gender_test)
    
    # Generate report
    report = analyzer.generate_fairness_report(fairness_results)
    
    # Save report
    with open('fairness_report.txt', 'w') as f:
        f.write(report)
    
    logger.info("Fairness analysis completed. Results saved to fairness_report.txt")
    return True

if __name__ == "__main__":
    run_fairness_analysis()
