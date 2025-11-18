import pandas as pd
import numpy as np

def find_label_column():
    print("🔍 Searching for label column...")
    
    data = pd.read_csv('/home/aldo/nids_capstone/data/archive/cicids2017_cleaned.csv')
    
    print(f"Dataset shape: {data.shape}")
    print(f"\nAll columns: {list(data.columns)}")
    
    # Common label column names in NIDS datasets
    common_labels = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min', 'Attack Type'
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
