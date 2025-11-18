
# Network Intrusion Detection: Comparative Analysis of ML and LLM Approaches

Senior Capstone Research Project | Computer Science | Network Security

## Project Overview

This research project conducts a systematic comparison of machine learning and large language model approaches for network intrusion detection. The study evaluates traditional machine learning methods against emerging LLM capabilities using the CICIDS2017 dataset to determine their respective efficacy in identifying network security threats.

## Current Status

The project has successfully implemented and evaluated a machine learning approach for network intrusion detection, with LLM integration currently in progress. The ML model has established a strong performance baseline for comparative analysis.

## Key Features

- Multi-method comparison across different AI paradigms
- Dockerized LLM environment for reproducible experimentation
- Comprehensive evaluation metrics including accuracy, precision, recall, and F1-score
- Feature importance analysis to identify predictive network characteristics
- Standardized testing framework for fair performance comparison

## Current Results

### Machine Learning Model (Random Forest)
- Accuracy: 99.87%
- Precision: 99.39%
- Recall: 99.84%
- F1-Score: 99.61%

Key finding: Packet length statistics and flow characteristics emerged as the most predictive features for intrusion detection.

### LLM Performance
- Status: Docker Ollama implementation complete, evaluation in progress
- Research focus: Assessing LLM capability to understand network patterns through natural language processing

PROJECT STRUCTURE
nids_capstone/
  src/
    train_ml.py (Machine learning training)
    train_llm.py (LLM evaluation)
    compare_results.py (Performance comparison)
    data_loader.py (Data preprocessing utilities)
    utils.py (Common functions)
  docker/
    ollama/
      docker-compose.yml
      start-ollama.sh
      stop-ollama.sh
  data/ (Dataset - gitignored)
    processed/
  models/ (Trained models - gitignored)
  results/ (Evaluation results and visualizations)
  docs/ (Documentation and research notes)


## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- Docker
- 8GB RAM minimum, 16GB recommended

### Installation

```bash
# Clone repository
git clone https://github.com/ALopezDoherty/nids_capstone.git
cd nids_capstone

# Create Python environment
python -m venv nids_env
source nids_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Docker Ollama
cd docker/ollama
chmod +x *.sh
./start-ollama.sh
cd ../..
```

### Basic Usage

```bash
# Train ML model (baseline)
python src/train_ml.py

# Evaluate LLM approach
python src/train_llm.py

# Compare results
python src/compare_results.py
```

## Research Methodology

1. **Data Preparation**: CICIDS2017 dataset preprocessing and feature creation
2. **Model Training**: Random Forest implementation with class balancing for dataset imbalance
3. **LLM Integration**: Dockerized Ollama deployment with structured prompt engineering
4. **Evaluation Framework**: Standardized metrics across all approaches
5. **Comparative Analysis**: Statistical performance comparison

## Technical Stack

- Machine Learning: Scikit-learn, Pandas, NumPy
- LLM Infrastructure: Ollama, Docker, REST API
- Visualization: Matplotlib, Seaborn
- Data Processing: Pandas, NumPy
- Version Control: Git, GitHub

## Research Implications

This study contributes to understanding the applicability of different AI approaches to network security. The comparison between specialized machine learning models and general-purpose language models provides insights for security practitioners and researchers regarding method selection based on accuracy requirements, computational constraints, and explainability needs.

## Acknowledgments

- CICIDS2017 dataset providers
- Professor O.
- Ollama and Docker communities for containerization support
```
