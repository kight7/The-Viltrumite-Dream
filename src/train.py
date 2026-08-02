import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, average_precision_score
from xgboost import XGBClassifier

def main():
    print("Loading data...")

    df = pd.read_csv('data/cleaned.csv')
    
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    
    df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
    X = df.drop(columns=['koi_disposition', 'target'])
    X = X.select_dtypes(include=[np.number])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Train XGBoost with scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.2f})...")
    
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED']))
    
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FALSE POSITIVE', 'CONFIRMED'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('docs/confusion_matrix.png')
    print("\nConfusion matrix saved to docs/confusion_matrix.png")
    
    # 5. Save model
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to models/xgb_model.pkl")

if __name__ == "__main__":
    main()
