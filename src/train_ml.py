import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

def train_final_model():
    print("FINAL ML TRAINING...")
    
    # Load data
    data = pd.read_csv('data/processed/cleaned_data.csv')
    print(f"Data loaded: {data.shape}")
    
    # === REPLACE THIS WITH YOUR ACTUAL LABEL COLUMN ===
    LABEL_COLUMN = 'Label'  # Change this based on find_label.py output
    
    if LABEL_COLUMN not in data.columns:
        print(f"Column '{LABEL_COLUMN}' not found. Available columns: {list(data.columns)}")
        return
    
    # Prepare features and target
    X = data.drop(LABEL_COLUMN, axis=1)
    y = data[LABEL_COLUMN]
    
    # If labels are text (like 'BENIGN', 'DoS'), convert to binary
    if y.dtype == 'object':
        print("Converting text labels to binary (0=normal, 1=attack)...")
        y_binary = (y != 'BENIGN').astype(int)  # Assuming 'BENIGN' means normal
        y = y_binary
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"FINAL RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save everything
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/final_random_forest.pkl')
    
    # Save results
    results = {
        'model': 'Random Forest',
        'accuracy': float(accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'label_column': LABEL_COLUMN,
        'timestamp': datetime.now().isoformat(),
        'top_features': feature_importance.head(10).to_dict('records')
    }
    
    with open('results/final_ml_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    plt.figure(figsize=(10, 6))
    
    # Confusion matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Feature importance
    plt.subplot(1, 2, 2)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/final_ml_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("FINAL ML training complete! Check 'results/' folder")
    return accuracy

if __name__ == "__main__":
    train_final_model()