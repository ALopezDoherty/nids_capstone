import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime
import time
import requests
import subprocess

class DockerOllamaAnalyzer:
    def __init__(self, model_name="llama3.1:8b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.setup_docker_ollama()
    
    def setup_docker_ollama(self):
        """Check if Docker Ollama is running"""
        print("Checking Docker Ollama setup...")
        
        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=nids-ollama", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if "nids-ollama" in result.stdout:
                print("Docker Ollama container is running")
                
                # Check API
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("Ollama API is accessible")
                    
                    # Check if model is available
                    models = response.json().get('models', [])
                    model_names = [model['name'] for model in models]
                    
                    if self.model_name in model_names:
                        print(f"Model '{self.model_name}' is available")
                    else:
                        print(f"Model '{self.model_name}' not found. Pulling...")
                        self.pull_model()
                else:
                    print("Ollama API not responding")
                    self.start_docker_ollama()
            else:
                print("Docker Ollama container not running. Starting...")
                self.start_docker_ollama()
                
        except Exception as e:
            print(f"Docker setup error: {e}")
            print("💡 Please run: ./docker/ollama/start-ollama.sh")
    
    def start_docker_ollama(self):
        """Start Docker Ollama using our script"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), '..', 'docker', 'ollama', 'start-ollama.sh')
            result = subprocess.run([script_path], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("✅ Docker Ollama started successfully")
                time.sleep(10)  # Wait for service to stabilize
            else:
                print(f"Failed to start Docker Ollama: {result.stderr}")
        except Exception as e:
            print(f"Error starting Docker Ollama: {e}")
    
    def pull_model(self):
        """Pull model via Docker"""
        try:
            print(f"Pulling model '{self.model_name}' via Docker...")
            result = subprocess.run(
                ["docker", "exec", "nids-ollama", "ollama", "pull", self.model_name],
                capture_output=True, text=True, timeout=300  # 5 minute timeout
            )
            if result.returncode == 0:
                print(f"Model '{self.model_name}' pulled successfully")
            else:
                print(f"Failed to pull model: {result.stderr}")
        except Exception as e:
            print(f"Error pulling model: {e}")
    
    def analyze_connection(self, prompt):
        """Analyze network connection using Docker Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 3  # We only need A or B
            }
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                answer = result['response'].strip().upper()
                
                # Extract just A or B from response
                if 'A' in answer and 'B' in answer:
                    return 'A' if answer.find('A') < answer.find('B') else 'B'
                elif 'B' in answer:
                    return 'B'
                else:
                    return 'A'
                    
            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
            except Exception as e:
                print(f"Docker Ollama API Error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        print("All retries failed. Using fallback response.")
        return 'A'  # Fallback to normal

def train_llm_docker():
    print("LLM TRAINING WITH DOCKER OLLAMA...")
    
    # Load data
    data = pd.read_csv('data/processed/cleaned_data.csv')
    print(f"Data loaded: {data.shape}")
    
    # Set label parameters
    LABEL_COLUMN = 'Attack Type'
    NORMAL_VALUE = 'Normal Traffic'
    
    if LABEL_COLUMN not in data.columns:
        print(f"Column '{LABEL_COLUMN}' not found. Available columns: {list(data.columns)}")
        return
    
    # Convert labels to binary
    y_true = (data[LABEL_COLUMN] != NORMAL_VALUE).astype(int)
    
    print(f"Dataset Info:")
    print(f"  Total samples: {len(data):,}")
    print(f"  Normal traffic: {(y_true == 0).sum():,} ({(y_true == 0).sum()/len(y_true)*100:.1f}%)")
    print(f"  Attack traffic: {(y_true == 1).sum():,} ({(y_true == 1).sum()/len(y_true)*100:.1f}%)")
    
    # Sample size - Docker might be faster due to better resource isolation
    sample_size = 40
    sample_data = data.sample(sample_size, random_state=42)
    sample_true = y_true[sample_data.index]
    
    print(f"\n Testing on {sample_size} samples...")
    
    # Initialize Docker Ollama analyzer
    ollama = DockerOllamaAnalyzer(model_name="llama3.1:8b")
    
    # Convert network data to natural language prompts
    def create_ollama_prompt(row):
        features = {
            'destination_port': row.get('Destination Port', 'N/A'),
            'flow_duration': f"{row.get('Flow Duration', 0) / 1000000:.2f}",
            'total_fwd_packets': row.get('Total Fwd Packets', 'N/A'),
            'total_bwd_packets': row.get('Total Bwd Packets', 'N/A'),
            'fwd_packet_length_max': row.get('Fwd Packet Length Max', 'N/A'),
            'flow_bytes_per_sec': f"{row.get('Flow Bytes/s', 0):.0f}",
            'flow_packets_per_sec': f"{row.get('Flow Packets/s', 0):.0f}",
        }
        
        prompt = f"""
        You are a network security expert. Analyze this network connection:

        Features:
        - Destination Port: {features['destination_port']}
        - Duration: {features['flow_duration']} seconds
        - Forward Packets: {features['total_fwd_packets']}
        - Backward Packets: {features['total_bwd_packets']}
        - Max Packet Size: {features['fwd_packet_length_max']} bytes
        - Bytes/Second: {features['flow_bytes_per_sec']}
        - Packets/Second: {features['flow_packets_per_sec']}

        Is this connection:
        A) Normal legitimate traffic
        B) Suspicious intrusion attempt

        Answer with only A or B:
        """
        
        return prompt
    
    # Generate predictions
    predictions = []
    actual_labels = []
    docker_responses = []
    
    print(f"\n Generating Docker Ollama predictions...")
    print(f"Model: {ollama.model_name}")
    print(f"Container: nids-ollama")
    
    total_start_time = time.time()
    
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        prompt = create_ollama_prompt(row)
        true_label = sample_true.loc[idx]
        
        # Get Ollama prediction
        start_time = time.time()
        pred_letter = ollama.analyze_connection(prompt)
        response_time = time.time() - start_time
        
        pred_binary = 1 if pred_letter == 'B' else 0
        actual_binary = int(true_label)
        
        predictions.append(pred_binary)
        actual_labels.append(actual_binary)
        
        docker_responses.append({
            'sample_id': int(idx),
            'true_label': actual_binary,
            'predicted_label': pred_binary,
            'predicted_letter': pred_letter,
            'correct': actual_binary == pred_binary,
            'response_time_seconds': round(response_time, 2),
            'model_used': ollama.model_name,
            'environment': 'docker'
        })
        
        print(f"  {i+1}/{sample_size}: True={actual_binary}, Pred={pred_binary} ({pred_letter}), Time={response_time:.1f}s")
        
        # Small delay
        if i < sample_size - 1:
            time.sleep(0.5)
    
    total_time = time.time() - total_start_time
    
    # Evaluate results
    accuracy = accuracy_score(actual_labels, predictions)
    
    print(f"\nDOCKER OLLAMA RESULTS:")
    print(f"Model: {ollama.model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Samples tested: {len(sample_data)}")
    print(f"Correct predictions: {sum(1 for r in docker_responses if r['correct'])}")
    print(f"Average response time: {np.mean([r['response_time_seconds'] for r in docker_responses]):.2f}s")
    print(f"Total execution time: {total_time:.1f}s")
    
    print("\nClassification Report:")
    print(classification_report(actual_labels, predictions, target_names=['Normal', 'Attack']))
    
    # Create results visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(actual_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Docker Ollama Confusion Matrix\n({ollama.model_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.subplot(1, 2, 2)
    correct = sum(1 for r in docker_responses if r['correct'])
    incorrect = len(docker_responses) - correct
    plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
    plt.title('Docker Ollama Prediction Accuracy')
    plt.ylabel('Number of Samples')
    
    for i, v in enumerate([correct, incorrect]):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/docker_ollama_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results = {
        'model': f'Docker Ollama ({ollama.model_name})',
        'accuracy': float(accuracy),
        'samples_tested': len(sample_data),
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'average_response_time': float(np.mean([r['response_time_seconds'] for r in docker_responses])),
        'total_execution_time': float(total_time),
        'environment': 'docker',
        'ollama_model': ollama.model_name,
        'timestamp': datetime.now().isoformat(),
        'sample_details': docker_responses
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/docker_ollama_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDocker Ollama testing complete!")
    print(f"Results saved to: results/docker_ollama_results.json")
    print(f"Plot saved to: results/docker_ollama_results.png")
    
    # Compare with other results
    comparison_data = []
    try:
        with open('results/final_ml_results.json', 'r') as f:
            ml_results = json.load(f)
        comparison_data.append(('ML Model', ml_results['accuracy'], ml_results['test_samples']))
    except:
        pass
    
    try:
        with open('results/ollama_results.json', 'r') as f:
            native_results = json.load(f)
        comparison_data.append(('Native Ollama', native_results['accuracy'], native_results['samples_tested']))
    except:
        pass
    
    if comparison_data:
        print(f"\nCOMPARISON WITH OTHER METHODS:")
        for name, acc, samples in comparison_data:
            print(f"  {name}: {acc:.4f} ({samples} samples)")
        print(f"  Docker Ollama: {accuracy:.4f} ({sample_size} samples)")
    
    return accuracy

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    train_llm_docker()