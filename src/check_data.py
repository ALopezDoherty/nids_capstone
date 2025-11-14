import pandas as pd
import os

def check_data_structure():
    # Try multiple possible data paths
    possible_paths = [
    '/home/aldo/nids_capstone/data/archive/cicids2017_cleaned.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found data at: {path}")
            data = pd.read_csv(path)
            break
    else:
        print("❌ No data file found. Check the paths above.")
        return
    
    print(f"\n📊 Data Shape: {data.shape}")
    print(f"\n📝 Column Names:")
    for i, col in enumerate(data.columns):
        print(f"  {i+1}. {col}")
    
    print(f"\n🔍 First 3 rows:")
    print(data.head(3))
    
    print(f"\n🎯 Looking for label/target column...")
    # Common label column names in intrusion detection datasets
    possible_label_columns = ['label', 'target', 'class', 'attack', 'intrusion', 'result']
    
    for col in possible_label_columns:
        if col in data.columns:
            print(f"✅ Found potential label column: '{col}'")
            print(f"   Value counts: {data[col].value_counts()}")
    
    return data

if __name__ == "__main__":
    check_data_structure()
