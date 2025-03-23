# Healthcare LLM Project: Diabetes Risk Prediction

This project demonstrates the application of Large Language Models (LLMs) to healthcare data, specifically for diabetes risk prediction. 

## Project Overview

- Processes synthetic patient data from Synthea
- Implements various LLM prompting techniques:
  - Basic prompting
  - In-context learning
  - Few-shot learning
  - Chain-of-thought reasoning
  - Tree-of-thought reasoning
  - Combined approach (for bonus implementation)
- Evaluates and compares the effectiveness of different methods

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd healthcare-llm-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the Synthea synthetic patient dataset, which provides realistic but non-real patient records.

## Usage

1. Download the Synthea dataset and place it in the `data/synthea` directory
2. Run the experiment notebook:
```bash
jupyter notebook notebooks/experiment.ipynb
```

## Results

[To be added after experiments]

## License

[Your License]
