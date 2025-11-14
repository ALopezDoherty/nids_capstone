import pandas as pd
import numpy as np

def find_label_column():
    print("🔍 Searching for label column...")
    
    data = pd.read_csv('data/processed/cleaned_data.csv')
    
    print(f"Dataset shape: {data.shape}")
    print(f"\nAll columns: {list(data.columns)}")
    
    # Common label column names in NIDS datasets
    common_labels = [
        'label', 'labels', 'target', 'class', 'classification',
        'attack', 'attack_type', 'attack_cat', 'category',
        'intrusion', 'malicious', 'benign', 'result', 'outcome',
        'Label', 'Labels', 'Target', 'Class'
    ]
    
    print(f"\n🎯 Looking for common label columns:")
    found = False
    for col in common_labels:
        if col in data.columns:
            print(f"✅ FOUND: '{col}'")
            print(f"   Unique values: {data[col].unique()}")
            print(f"   Value counts:\n{data[col].value_counts()}")
            found = True
    
    if not found:
        print("❌ No common label columns found.")
        
        # Check for binary columns (often used as labels)
        print(f"\n🔍 Checking for binary columns (potential labels):")
        for col in data.columns:
            unique_vals = data[col].unique()
            if len(unique_vals) == 2:
                print(f"📊 Binary column '{col}': {unique_vals}")
                print(f"   Value counts:\n{data[col].value_counts()}")
        
        # Check for categorical columns with few values
        print(f"\n🔍 Checking categorical columns:")
        for col in data.columns:
            if data[col].dtype == 'object' and len(data[col].unique()) < 10:
                print(f"📊 Categorical column '{col}': {data[col].unique()}")
    
    # Show data types
    print(f"\n📊 Data types:")
    print(data.dtypes)

if __name__ == "__main__":
    find_label_column()