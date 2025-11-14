import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from datetime import datetime

def train_ml_model():
    print("🚀 TRAINING ML MODEL...")
    
    # 1. Load your data
    try:
        data = pd.read_csv('data/processed/cleaned_data.csv')
        print(f"✅ Data loaded: {data.shape}")
    except:
        print("❌ Could not load data. Check file path.")
        return
    
    # 2. Prepare features and target
    # ADAPT THESE based on your actual data
    X = data.drop('label', axis=1)  # Features
    y = data['label']  # Target (0=normal, 1=intrusion)
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train model
    print("🤖 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f" ML Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save results
    results = {
        'model': 'Random Forest',
        'accuracy': float(accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': datetime.now().isoformat(),
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest.pkl')
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/ml_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('ML Model - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('results/ml_confusion_matrix.png')
    plt.close()
    
    print("✅ ML training complete! Results saved to results/")
    return accuracy

if __name__ == "__main__":
    train_ml_model()