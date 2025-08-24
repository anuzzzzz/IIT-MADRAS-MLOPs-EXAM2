import sys
sys.path.append('src')
from src.model_training import HeartDiseasePredictor
import pandas as pd
import numpy as np

# Simple explainability analysis
predictor = HeartDiseasePredictor()
X, y, df = predictor.load_and_preprocess_data('data/data.csv')
predictor.load_model('models')

# Get heart disease samples
disease_samples = df[df['target'] == 1].head(3)
X_samples = disease_samples.drop(['target', 'sno'], axis=1, errors='ignore')

# Make predictions and explain
explanations = []
for i, (idx, row) in enumerate(X_samples.iterrows()):
    sample_df = pd.DataFrame([row])
    pred, prob = predictor.predict(sample_df)
    
    explanation = f"Sample {i+1}:\n"
    explanation += f"Prediction: {'Heart Disease' if pred[0] == 1 else 'No Heart Disease'}\n"
    explanation += f"Confidence: {prob[0][pred[0]]:.2%}\n"
    explanation += f"Age: {row['age']}, Gender: {'Male' if row['gender']==0 else 'Female'}\n"
    explanation += f"Chest Pain Type: {row['cp']}, Max Heart Rate: {row['thalach']}\n\n"
    explanations.append(explanation)

with open('heart_disease_explanations.txt', 'w') as f:
    f.write("HEART DISEASE PREDICTION EXPLANATIONS\n" + "="*50 + "\n\n")
    for exp in explanations:
        f.write(exp + "-"*30 + "\n")

print("✅ Explainability analysis completed")
