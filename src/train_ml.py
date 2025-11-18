import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime

def train_final_model():
    print("FINAL ML TRAINING...")
    
    # Load data
    data = pd.read_csv('/home/aldo/nids_capstone/data/archive/cicids2017_cleaned.csv')
    print(f"Data loaded: {data.shape}")
    
    LABEL_COLUMN = 'Attack Type' 
    NORMAL_VALUE = 'Normal Traffic'

    if LABEL_COLUMN not in data.columns:
        print(f"Column '{LABEL_COLUMN}' not found. Available columns: {list(data.columns)}")
        return
    
    # Prepare features and target
    X = data.drop(LABEL_COLUMN, axis=1)
    y = data[LABEL_COLUMN]

    # Convert text labels to binary (0=normal, 1=attack)
    print("Converting text labels to binary (0=normal, 1=attack)...")
    print(f"Before conversion: {y.value_counts()}")
    
    y_binary = (y != NORMAL_VALUE).astype(int)  # This converts 'Normal Traffic' to 0, everything else to 1
    y = y_binary
    
    print(f"After conversion: ")
    binary_counts = pd.Series(y).value_counts()
    print(f"  0 = Normal Traffic")
    print(f"  1 = All Attacks (Port Scanning, Web Attacks, Brute Force, DDoS, Bots, DoS)")

    # Check for class imbalance
    normal_count = binary_counts.get(0, 0)
    attack_count = binary_counts.get(1, 0)
    total_count = len(y)
    
    print(f"\n📊 Class Distribution:")
    print(f"  Normal traffic: {normal_count} ({normal_count/total_count*100:.1f}%)")
    print(f"  Attack traffic: {attack_count} ({attack_count/total_count*100:.1f}%)")
    
    if normal_count == 0 or attack_count == 0:
        print("❌ ERROR: Only one class found in data! Cannot train binary classifier.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📈 Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Calculate class weights to handle imbalance
    try:
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print("⚖️ Using class weights to handle imbalance:")
        print(f"  Class 0 (normal) weight: {class_weight_dict[0]:.2f}")
        print(f"  Class 1 (attack) weight: {class_weight_dict[1]:.2f}")
    except:
        print("⚠️  Using default class weights")
        class_weight_dict = 'balanced'
    
    # Train model with class weights
    print("🤖 Training Random Forest with class weights...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight_dict
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 FINAL RESULTS:")
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
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save model
    model_path = 'models/final_random_forest.pkl'
    joblib.dump(model, model_path)
    
    # Save results
    results = {
        'model': 'Random Forest',
        'accuracy': float(accuracy),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'label_column': LABEL_COLUMN,
        'normal_value': NORMAL_VALUE,
        'class_distribution': {
            'normal_count': int(normal_count),
            'attack_count': int(attack_count),
            'normal_percentage': float(normal_count/total_count*100),
            'attack_percentage': float(attack_count/total_count*100)
        },
        'timestamp': datetime.now().isoformat(),
        'top_features': feature_importance.head(10).to_dict('records')
    }
    
    with open('results/final_ml_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    # Confusion matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix\n(0=Normal, 1=Attack)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Feature importance
    plt.subplot(1, 2, 2)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/final_ml_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n FINAL ML training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: results/final_ml_results.json")
    print(f"Plot saved to: results/final_ml_results.png")
    
    # Final summary
    print(f"\nSUMMARY:")
    print(f"  - Test Accuracy: {accuracy:.4f}")
    print(f"  - Training Samples: {len(X_train):,}")
    print(f"  - Test Samples: {len(X_test):,}")
    print(f"  - Most Important Feature: {feature_importance.iloc[0]['feature']}")
    
    return accuracy

if __name__ == "__main__":
    train_final_model()