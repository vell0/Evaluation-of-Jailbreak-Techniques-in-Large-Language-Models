# Evaluation of Jailbreak Techniques in Large Language Models

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Dataset: Huggingface](https://img.shields.io/badge/Dataset-Huggingface-yellow.svg)](https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the thesis "Jailbreaking Techniques for LLMs: A Systematic Evaluation Framework" that introduces a unified evaluation framework for comparing different jailbreaking methodologies across multiple model architectures and scales.

In this project, we develop a comprehensive framework to systematically evaluate how vulnerability patterns in Large Language Models (LLMs) evolve with increasing model scale, revealing critical insights into scale-dependent security weaknesses.


**Disclaimer: This repository contains examples of harmful language. Reader discretion is recommended. This repo is intended for research purposes only. Any misuse is strictly prohibited.**

## Key Findings

Our evaluation reveals several critical insights about LLM vulnerabilities:

1. **Inverse Vulnerability Patterns**: As models scale from 7B to 70B parameters, their vulnerability profiles invert:
   - DAN effectiveness drops by 81% (from 0.69 to 0.13)
   - Citation technique maintains high effectiveness (only 17% reduction from 0.59 to 0.49)

2. **Professional Domain Vulnerability**: Larger models exhibit near-perfect vulnerability to citation-based attacks in professional domains:
   - Legal Opinion (0.99)
   - Financial Advice (1.00)
   - Health Consultation (0.99)
   - Political Lobbying (1.00)

3. **Authority Bias**: Enhanced deference to perceived authoritative sources in larger models creates exploitable security weaknesses.

4. **Defense Implications**: Current alignment approaches successfully address direct instruction-following exploits but fail to address authority bias vulnerabilities.

## Repository Structure

```
.
├── main_code.py                    # Main implementation with UnifiedEvaluator class
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License file
├── datasets/                       # Dataset scripts (not datasets themselves)
│   ├── README.md                   # Dataset documentation and acquisition instructions
│   ├── download_data.py            # Script to download from official sources
│   ├── process_dan_prompts.py      # Script for DAN prompt selection and filtering
│   └── citation_generator.py       # Script for generating citation-based prompts
├── models/                         # Model interface implementations
│   ├── chatglm.py                  # ChatGLM interface for evaluation model
│   ├── vicuna.py                   # Vicuna model interface
│   ├── llama.py                    # Llama model interface
│   ├── mistral.py                  # Mistral model interface
│   └── gemma.py                    # Gemma model interface
├── utils/                          # Utility functions and helpers
│   ├── common.py                   # Shared utility functions
│   ├── config.py                   # Configuration utilities
│   └── system_template.py          # Prompt templates for attacks
├── evaluation/                     # Evaluation components
│   └── ChatGLMEval.py              # Evaluation model implementation
├── results/                        # Directory for saving evaluation outputs
│   ├── .gitignore                  # Excludes actual results from repository
│   └── README.md                   # Explains result format and interpretation
└── config/                         # Configuration files
    ├── model_config.json           # Model configurations
    └── evaluation_config.json      # Evaluation parameters
```
## Dataset Information

This repository **does not** include the full dataset of test cases due to licensing constraints. Instead, we provide:

1. Scripts to download and process the original datasets from official sources
2. Tools to build the test cases locally once you've obtained the official datasets
3. Documentation on the dataset structure and composition

The datasets required for reproduction include:

- **TrustAIRLab jailbreak prompts dataset** (MIT License) - [Hugging Face Link](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts)
- **TrustAIRLab forbidden question set** - [Hugging Face Link](https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set)

To reproduce our results:
1. Download the official datasets from the links above
2. Run `datasets/download_data.py` to automatically fetch these datasets
3. Run `datasets/process_dan_prompts.py` to select and filter DAN prompts
4. Use the framework for evaluation as described below

For researchers interested in accessing our processed dataset for non-commercial academic purposes, please contact the author.

## Jailbreak Techniques

### DAN Technique

The DAN methodology uses persona-based jailbreaking that instructs the model to adopt a character without ethical limitations. Our implementation includes:

1. Selecting 30 representative DAN prompts from curated dataset
2. Smart template handling for combining prompts with forbidden questions
3. Systematic evaluation across multiple harm categories

### DarkCite Technique

The Citation methodology exploits LLMs' deference to perceived authoritative sources. Our implementation includes:

1. Dynamic generation of domain-appropriate citations
2. Category-specific citation mapping (e.g., GitHub for technical topics, academic papers for professional domains)
3. Structured prompt construction that establishes scholarly context

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-jailbreak-framework.git
cd llm-jailbreak-framework

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Requirements

Python 3.10+
PyTorch 2.0.1
Transformers 4.31.0
CUDA 11.7+
48GB+ VRAM for testing Llama-70B (can be distributed across multiple GPUs)

Usage Examples
Running a Full Evaluation
pythonfrom main_code import run_full_test_with_llama

# Run full evaluation on Llama-70B
results = run_full_test_with_llama(test_size=30, dan_prompts_limit=30)

# Print results summary
for scenario, metrics in results.items():
    print(f"{scenario}:")
    print(f"  DAN ASR: {metrics['DAN']['ASR']:.3f}")
    print(f"  DarkCite ASR: {metrics['DarkCite']['ASR']:.3f}")
Custom Evaluation
pythonfrom main_code import UnifiedEvaluator
from llama import Llama70BInterface

# Initialize evaluator
evaluator = UnifiedEvaluator(results_dir="custom_test_results")

# Load model
llm = Llama70BInterface()

# Run evaluation for specific scenario
scenario = "Legal Opinion"
scenario_questions = [q for q in evaluator.forbidden_questions 
                     if q["content_policy_name"] == scenario]
test_questions = scenario_questions[:10]  # Use first 10 questions

# Evaluate methodologies
dan_results = evaluator.evaluate_dan(test_questions, llm)
darkcite_results = evaluator.evaluate_darkcite(test_questions, llm, scenario)

# Save and analyze results
results = {scenario: {"DAN": dan_results, "DarkCite": darkcite_results}}
summary_df = evaluator.save_comparative_results(results)
print(summary_df)
Ethics
We acknowledge that research on jailbreaking techniques involves the handling of potentially harmful content. Our work follows strict ethical principles:

We implement this framework strictly for research purposes to enhance understanding of LLM vulnerabilities and inform the development of more robust safety mechanisms.
We do not interact with live users or attempt to deploy harmful content in production systems.
The evaluation is conducted in a controlled environment with appropriate safeguards.
Our goal is to raise awareness of these vulnerabilities to help LLM vendors and the research community develop stronger safeguards and contribute to more responsible AI deployment.

We have responsibly disclosed our findings to relevant LLM vendors.

# Install dependencies
pip install -r requirements.txt
```

