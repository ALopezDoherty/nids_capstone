import json
import matplotlib.pyplot as plt
import numpy as np

def compare_models():
    print("📈 COMPARING MODEL RESULTS...")
    
    # Load results
    with open('results/ml_results.json', 'r') as f:
        ml_results = json.load(f)
    
    with open('results/llm_results.json', 'r') as f:
        llm_results = json.load(f)
    
    # Create comparison bar chart
    models = ['Random Forest', 'LLM (Mock)']
    accuracies = [ml_results['accuracy'], llm_results['accuracy']]
    samples = [ml_results['test_samples'], llm_results['samples_tested']]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    bars = ax1.bar(models, accuracies, color=['blue', 'orange'], alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Sample size comparison
    ax2.bar(models, samples, color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Test Sample Sizes')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*50)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Random Forest ML Model:")
    print(f"  - Accuracy: {ml_results['accuracy']:.4f}")
    print(f"  - Test Samples: {ml_results['test_samples']}")
    print(f"  - Top Features: {list(ml_results['feature_importance'].keys())[:3]}")
    
    print(f"\nLLM Approach (Mock):")
    print(f"  - Accuracy: {llm_results['accuracy']:.4f}")
    print(f"  - Test Samples: {llm_results['samples_tested']}")
    
    print(f"\n✅ Comparison complete! Check 'results/model_comparison.png'")

if __name__ == "__main__":
    compare_models()