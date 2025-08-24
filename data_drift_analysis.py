import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from src.model_training import HeartDiseasePredictor

def simple_drift_analysis():
    """Simple data drift analysis comparing training vs test data"""
    print("🔍 STARTING DATA DRIFT ANALYSIS")
    print("=" * 40)
    
    # Load original training data
    predictor = HeartDiseasePredictor()
    X_original, y_original, df_original = predictor.load_and_preprocess_data('data/data.csv')
    
    # Load generated test data
    df_generated = pd.read_csv('random_test_data_100.csv')
    
    print(f"📊 Original data shape: {X_original.shape}")
    print(f"📊 Generated data shape: {df_generated.shape}")
    
    # Compare distributions
    drift_report = "DATA DRIFT ANALYSIS REPORT\n"
    drift_report += "=" * 40 + "\n\n"
    
    drift_report += "STATISTICAL COMPARISON:\n"
    drift_report += "-" * 25 + "\n"
    
    significant_drift_features = []
    
    for feature in X_original.columns:
        if feature in df_generated.columns:
            original_mean = X_original[feature].mean()
            generated_mean = df_generated[feature].mean()
            original_std = X_original[feature].std()
            generated_std = df_generated[feature].std()
            
            # Calculate drift score (normalized difference)
            mean_drift = abs(original_mean - generated_mean) / original_std if original_std > 0 else 0
            std_drift = abs(original_std - generated_std) / original_std if original_std > 0 else 0
            
            drift_score = (mean_drift + std_drift) / 2
            
            drift_report += f"\n{feature}:\n"
            drift_report += f"  Original: μ={original_mean:.2f}, σ={original_std:.2f}\n"
            drift_report += f"  Generated: μ={generated_mean:.2f}, σ={generated_std:.2f}\n"
            drift_report += f"  Drift Score: {drift_score:.3f}\n"
            
            if drift_score > 0.2:  # Threshold for significant drift
                drift_report += f"  ⚠️  SIGNIFICANT DRIFT DETECTED\n"
                significant_drift_features.append(feature)
            else:
                drift_report += f"  ✅ No significant drift\n"
    
    drift_report += f"\nSUMMARY:\n"
    drift_report += f"Features with significant drift: {len(significant_drift_features)}\n"
    drift_report += f"Drifted features: {significant_drift_features}\n\n"
    
    if len(significant_drift_features) > 0:
        drift_report += "⚠️  DATA DRIFT DETECTED!\n"
        drift_report += "Recommendations:\n"
        drift_report += "- Monitor model performance closely\n"
        drift_report += "- Consider retraining with recent data\n"
        drift_report += "- Implement drift detection in production\n"
    else:
        drift_report += "✅ NO SIGNIFICANT DRIFT\n"
        drift_report += "Generated data is similar to training distribution\n"
    
    # Save report
    with open('data_drift_report.txt', 'w') as f:
        f.write(drift_report)
    
    print(drift_report)
    print("📄 Drift analysis saved to data_drift_report.txt")

if __name__ == "__main__":
    simple_drift_analysis()
