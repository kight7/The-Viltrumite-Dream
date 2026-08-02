import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def main():
    print("Loading test data...")
    df = pd.read_csv('data/cleaned.csv')
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
    
    X = df.drop(columns=['koi_disposition', 'target'])
    X = X.select_dtypes(include=[np.number])
    y = df['target']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    sample_X = X_test.head(10)
    sample_y = y_test.head(10)
    
    print("Loading model from models/xgb_model.pkl...")
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    print("\n--- Predictions on 10 Held-Out Samples ---")
    predictions = model.predict(sample_X)
    probabilities = model.predict_proba(sample_X)[:, 1]
    
    label_map = {1: 'CONFIRMED', 0: 'FALSE POSITIVE'}
    
    for i in range(len(sample_y)):
        true_val = sample_y.iloc[i]
        pred_val = predictions[i]
        prob = probabilities[i]
        
        print(f"Row {i+1}:")
        print(f"  True Label:      {label_map[true_val]}")
        print(f"  Predicted Label: {label_map[pred_val]}")
        print(f"  Probability of CONFIRMED: {prob:.4f}\n")

if __name__ == "__main__":
    main()
