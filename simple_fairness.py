import sys
sys.path.append('src')
from src.model_training import HeartDiseasePredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Simple fairness analysis
predictor = HeartDiseasePredictor()
X, y, df = predictor.load_and_preprocess_data('data/data.csv')
predictor.load_model('models')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Get predictions
predictions, _ = predictor.predict(X_test)
gender_test = X_test['gender'].values

# Analyze by gender
male_mask = gender_test == 0
female_mask = gender_test == 1

male_accuracy = accuracy_score(y_test[male_mask], predictions[male_mask])
female_accuracy = accuracy_score(y_test[female_mask], predictions[female_mask])

report = f"""FAIRNESS ANALYSIS REPORT
========================================

GENDER-BASED PERFORMANCE:
Male accuracy: {male_accuracy:.4f} (n={sum(male_mask)})
Female accuracy: {female_accuracy:.4f} (n={sum(female_mask)})
Accuracy difference: {abs(male_accuracy - female_accuracy):.4f}

INTERPRETATION:
{'✓ Fair performance across genders' if abs(male_accuracy - female_accuracy) < 0.1 else '⚠ Potential gender bias detected'}
"""

with open('fairness_report.txt', 'w') as f:
    f.write(report)

print("✅ Fairness analysis completed")
