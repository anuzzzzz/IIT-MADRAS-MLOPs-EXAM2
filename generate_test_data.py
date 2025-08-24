import pandas as pd
import numpy as np
import json

def generate_random_heart_data(n_samples=100):
    """Generate 100 random heart disease samples for testing"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        sample = {
            'age': int(np.random.randint(25, 80)),
            'gender': int(np.random.choice([0, 1])),
            'cp': int(np.random.randint(0, 4)),
            'trestbps': float(np.round(np.random.uniform(90, 200), 1)),
            'chol': float(np.round(np.random.uniform(120, 400), 1)),
            'fbs': int(np.random.choice([0, 1])),
            'restecg': int(np.random.randint(0, 3)),
            'thalach': float(np.round(np.random.uniform(60, 200), 1)),
            'exang': int(np.random.choice([0, 1])),
            'oldpeak': float(np.round(np.random.uniform(0, 6), 1)),
            'slope': int(np.random.randint(0, 3)),
            'ca': int(np.random.randint(0, 4)),
            'thal': int(np.random.randint(0, 4))
        }
        data.append(sample)
    
    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv('random_test_data_100.csv', index=False)
    
    # Save for batch API testing
    with open('batch_test_samples.json', 'w') as f:
        json.dump({'samples': data}, f, indent=2)
    
    print(f"✅ Generated {n_samples} test samples")
    return data

if __name__ == "__main__":
    generate_random_heart_data(100)
