import pandas as pd
import numpy as np
import json

def generate_100_random_samples():
    """Generate exactly 100 random heart disease samples"""
    np.random.seed(42)  # For reproducibility
    
    samples = []
    for i in range(100):
        sample = {
            'sample_id': i + 1,
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
        samples.append(sample)
    
    # Save as CSV
    df = pd.DataFrame(samples)
    df.to_csv('100_random_samples.csv', index=False)
    
    # Save individual samples for wrk testing
    with open('sample_for_wrk.json', 'w') as f:
        json.dump(samples[0], f, indent=2)  # First sample for wrk
    
    print(f"✅ Generated exactly 100 random samples")
    print(f"📁 Saved to: 100_random_samples.csv")
    print(f"📁 Sample for wrk: sample_for_wrk.json")
    
    return samples

if __name__ == "__main__":
    generate_100_random_samples()
