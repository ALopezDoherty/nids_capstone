import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime

def train_llm():
    print("LLM TESTING...")
    
    # Load data
    data = pd.read_csv('/home/aldo/nids_capstone/data/archive/cicids2017_cleaned.csv')
    
    # Use the same label column as ML script
    LABEL_COLUMN = 'Label'  # Change this to match your label column
    
    # Take a sample for testing
    sample_data = data.sample(100, random_state=42)
    
    # Convert to text descriptions
    def create_llm_prompt(row):
        # Create a natural language description
        prompt = f"""
        Analyze this network connection for intrusion detection:
        
        Connection Features:
        - Protocol: {row.get('Protocol', 'N/A')}
        - Duration: {row.get('Duration', 'N/A')} seconds
        - Source Bytes: {row.get('Src Bytes', 'N/A')}
        - Destination Bytes: {row.get('Dst Bytes', 'N/A')}
        - Source Packets: {row.get('Tot Fwd Pkts', 'N/A')}
        - Destination Packets: {row.get('Tot Bwd Pkts', 'N/A')}
        
        Based on these network traffic patterns, classify this connection as either:
        A) Normal legitimate traffic
        B) Suspicious intrusion attempt
        
        Provide only your final classification (A or B):
        """
        return prompt
    
    # mock LLM with better heuristics
    def llm_prediction(prompt, row):
        """
        Better rule-based mock. Replace with real LLM API later.
        """
        # Simple heuristics based on common attack patterns
        src_bytes = row.get('Src Bytes', 0)
        dst_bytes = row.get('Dst Bytes', 0)
        duration = row.get('Duration', 0)
        
        # Rule 1: Very short duration with high bytes = possible DoS
        if duration < 0.1 and (src_bytes > 1000 or dst_bytes > 1000):
            return 'B'
        # Rule 2: Zero source bytes but destination bytes = possible scanning
        elif src_bytes == 0 and dst_bytes > 0:
            return 'B'
        # Rule 3: Many source packets in short time = possible flooding
        elif row.get('Tot Fwd Pkts', 0) > 100 and duration < 1.0:
            return 'B'
        else:
            return 'A'
    
    # Generate predictions
    predictions = []
    actual_labels = []
    llm_responses = []
    
    print("Generating LLM predictions...")
    for idx, row in sample_data.iterrows():
        prompt = create_llm_prompt(row)
        true_label = row[LABEL_COLUMN]
        
        # Convert true label to binary (0=normal, 1=attack)
        if true_label == 'BENIGN' or true_label == 0:
            true_binary = 0
        else:
            true_binary = 1
        
        # Get LLM prediction
        pred_letter = llm_prediction(prompt, row)
        pred_binary = 1 if pred_letter == 'B' else 0
        
        predictions.append(pred_binary)
        actual_labels.append(true_binary)
        
        llm_responses.append({
            'true_label': str(true_label),
            'true_binary': true_binary,
            'predicted_binary': pred_binary,
            'correct': true_binary == pred_binary
        })
    
    # Evaluate
    accuracy = accuracy_score(actual_labels, predictions)
    
    print(f"LLM RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Samples tested: {len(sample_data)}")
    print("\nClassification Report:")
    print(classification_report(actual_labels, predictions))
    
    # Save results
    results = {
        'model': 'LLM (Mock)',
        'accuracy': float(accuracy),
        'samples_tested': len(sample_data),
        'timestamp': datetime.now().isoformat(),
        'correct_predictions': sum(1 for r in llm_responses if r['correct']),
        'total_predictions': len(llm_responses)
    }
    
    with open('results/llm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("LLM testing complete!")
    return accuracy

if __name__ == "__main__":
    train_llm()