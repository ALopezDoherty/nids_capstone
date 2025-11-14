import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from datetime import datetime

def test_llm_approach():
    print("🧠 TESTING LLM APPROACH...")
    
    # 1. Load data
    data = pd.read_csv('data/processed/cleaned_data.csv')
    
    # 2. Take a small sample for quick testing (LLMs are slow/expensive)
    sample_data = data.sample(50, random_state=42)  # Small sample for testing
    
    # 3. Convert network data to text descriptions
    def create_prompt(row):
        # Create a text description of the network connection
        # ADAPT based on your actual column names
        prompt = f"""
        Analyze this network connection:
        - Protocol: {row.get('protocol_type', 'N/A')}
        - Duration: {row.get('duration', 'N/A')} seconds
        - Source Bytes: {row.get('src_bytes', 'N/A')}
        - Destination Bytes: {row.get('dst_bytes', 'N/A')}
        - Connection State: {row.get('conn_state', 'N/A')}
        
        Based on these features, is this likely to be:
        A) Normal network traffic
        B) Suspicious intrusion attempt
        
        Answer with only A or B:
        """
        return prompt
    
    # 4. Mock LLM function (replace with real API later)
    def mock_llm_prediction(prompt, true_label):
        """
        Simple rule-based mock. Replace this with:
        - OpenAI API
        - Hugging Face API  
        - Local LLM
        """
        # Simple heuristic rules (you'll replace this)
        if 'src_bytes' in prompt and int(prompt.split('Source Bytes: ')[1].split()[0]) > 1000:
            return 'B'  # intrusion
        elif 'duration' in prompt and float(prompt.split('Duration: ')[1].split()[0]) < 0.1:
            return 'B'  # intrusion
        else:
            return 'A'  # normal
    
    # 5. Generate predictions
    predictions = []
    llm_responses = []
    
    print("Generating LLM predictions...")
    for idx, row in sample_data.iterrows():
        prompt = create_prompt(row)
        true_label = row['label']
        
        # Convert true label to A/B format
        true_letter = 'B' if true_label == 1 else 'A'
        
        # Get LLM prediction
        pred_letter = mock_llm_prediction(prompt, true_label)
        predictions.append(1 if pred_letter == 'B' else 0)
        
        llm_responses.append({
            'prompt': prompt,
            'true_label': int(true_label),
            'predicted_label': 1 if pred_letter == 'B' else 0,
            'true_letter': true_letter,
            'pred_letter': pred_letter
        })
    
    # 6. Evaluate
    accuracy = accuracy_score(sample_data['label'], predictions)
    
    print(f"📊 LLM Results (Mock):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Samples tested: {len(sample_data)}")
    print("\nClassification Report:")
    print(classification_report(sample_data['label'], predictions))
    
    # 7. Save results
    results = {
        'model': 'LLM (Mock)',
        'accuracy': float(accuracy),
        'samples_tested': len(sample_data),
        'timestamp': datetime.now().isoformat(),
        'responses': llm_responses
    }
    
    with open('results/llm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ LLM testing complete! Results saved to results/")
    return accuracy

if __name__ == "__main__":
    test_llm_approach()