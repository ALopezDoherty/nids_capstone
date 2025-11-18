import json
import matplotlib.pyplot as plt
import numpy as np
import os

def compare_all_models():
    print("COMPARING ALL MODEL RESULTS...")
    
    results = []
    
    # Load ML results
    try:
        with open('results/final_ml_results.json', 'r') as f:
            ml_data = json.load(f)
        results.append({
            'name': 'Random Forest ML',
            'accuracy': ml_data['accuracy'],
            'samples': ml_data['test_samples'],
            'type': 'ml'
        })
    except FileNotFoundError:
        print("ML results not found")
    
    # Load Docker Ollama results
    try:
        with open('results/docker_ollama_results.json', 'r') as f:
            docker_data = json.load(f)
        results.append({
            'name': 'Docker Ollama',
            'accuracy': docker_data['accuracy'],
            'samples': docker_data['samples_tested'],
            'type': 'llm'
        })
    except FileNotFoundError:
        print("Docker Ollama results not found")
    
    # Load native Ollama results (if exists)
    try:
        with open('results/ollama_results.json', 'r') as f:
            native_data = json.load(f)
        results.append({
            'name': 'Native Ollama',
            'accuracy': native_data['accuracy'],
            'samples': native_data['samples_tested'],
            'type': 'llm'
        })
    except FileNotFoundError:
        pass
    
    if not results:
        print("No results found to compare")
        return
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    colors = ['blue' if r['type'] == 'ml' else 'orange' for r in results]
    
    bars = ax1.bar(names, accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Sample size comparison
    samples = [r['samples'] for r in results]
    ax2.bar(names, samples, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Test Samples')
    ax2.set_title('Test Sample Sizes')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(samples):
        ax2.text(i, v + max(samples)*0.01, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/final_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['name']:>20}: {result['accuracy']:.4f} ({result['samples']:,} samples)")
    
    # Save comparison results
    comparison = {
        'timestamp': str(np.datetime64('now')),
        'models_compared': len(results),
        'results': results,
        'best_model': max(results, key=lambda x: x['accuracy'])['name'],
        'best_accuracy': max(results, key=lambda x: x['accuracy'])['accuracy']
    }
    
    with open('results/model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison complete! Check 'results/final_comparison.png'")

if __name__ == "__main__":
    compare_all_models()