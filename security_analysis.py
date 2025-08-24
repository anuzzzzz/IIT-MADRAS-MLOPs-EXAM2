import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from src.model_training import HeartDiseasePredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def data_poisoning_analysis():
    """Simulate data poisoning attack and analyze impact"""
    print("🛡️  STARTING SECURITY ANALYSIS")
    print("=" * 40)
    
    # Load original data
    predictor = HeartDiseasePredictor()
    X, y, df = predictor.load_and_preprocess_data('data/data.csv')
    
    print(f"📊 Original dataset size: {len(X)}")
    
    # Create poisoned dataset (flip 15% of labels)
    poison_ratio = 0.15
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    
    # Randomly select samples to poison
    np.random.seed(42)
    n_poison = int(len(y) * poison_ratio)
    poison_indices = np.random.choice(len(y), n_poison, replace=False)
    
    # Flip labels
    y_poisoned.iloc[poison_indices] = 1 - y_poisoned.iloc[poison_indices]
    
    print(f"🔴 Poisoned {n_poison} samples ({poison_ratio:.1%} of data)")
    
    # Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train original model
    print("🤖 Training clean model...")
    clean_predictor = HeartDiseasePredictor()
    clean_predictor.train_model(X_train, y_train)
    
    # Train poisoned model
    print("☠️  Training poisoned model...")
    poisoned_predictor = HeartDiseasePredictor()
    
    # Get poisoned training data
    X_train_poisoned = X_poisoned.loc[X_train.index]
    y_train_poisoned = y_poisoned.loc[X_train.index]
    
    poisoned_predictor.train_model(X_train_poisoned, y_train_poisoned)
    
    # Evaluate both models
    clean_pred, _ = clean_predictor.predict(X_test)
    poisoned_pred, _ = poisoned_predictor.predict(X_test)
    
    clean_accuracy = accuracy_score(y_test, clean_pred)
    poisoned_accuracy = accuracy_score(y_test, poisoned_pred)
    
    accuracy_drop = clean_accuracy - poisoned_accuracy
    agreement = np.mean(clean_pred == poisoned_pred)
    
    # Generate security report
    security_report = "SECURITY ANALYSIS REPORT\n"
    security_report += "=" * 40 + "\n\n"
    
    security_report += "DATA POISONING ATTACK SIMULATION:\n"
    security_report += f"Attack Type: Label Flipping\n"
    security_report += f"Poison Ratio: {poison_ratio:.1%}\n"
    security_report += f"Poisoned Samples: {n_poison}/{len(y)}\n\n"
    
    security_report += "PERFORMANCE IMPACT:\n"
    security_report += f"Clean Model Accuracy: {clean_accuracy:.4f}\n"
    security_report += f"Poisoned Model Accuracy: {poisoned_accuracy:.4f}\n"
    security_report += f"Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop/clean_accuracy:.1%})\n"
    security_report += f"Model Agreement: {agreement:.4f}\n\n"
    
    security_report += "ATTACK SUCCESS ASSESSMENT:\n"
    if accuracy_drop > 0.05:
        security_report += "🚨 SUCCESSFUL ATTACK - Significant performance degradation\n"
        security_report += "Model is vulnerable to data poisoning attacks\n\n"
        security_report += "SECURITY RECOMMENDATIONS:\n"
        security_report += "- Implement robust data validation\n"
        security_report += "- Use anomaly detection for training data\n"
        security_report += "- Monitor model performance continuously\n"
        security_report += "- Consider using robust training algorithms\n"
        security_report += "- Implement data provenance tracking\n"
    else:
        security_report += "🛡️  RESILIENT MODEL - Limited attack impact\n"
        security_report += "Model shows good robustness against label flipping\n\n"
        security_report += "SECURITY STATUS: ACCEPTABLE\n"
        security_report += "- Continue monitoring for sophisticated attacks\n"
        security_report += "- Maintain data quality controls\n"
    
    # Save report
    with open('security_analysis_report.txt', 'w') as f:
        f.write(security_report)
    
    print(security_report)
    print("📄 Security analysis saved to security_analysis_report.txt")

if __name__ == "__main__":
    data_poisoning_analysis()
