# Evaluation-of-Jailbreak-Techniques-in-Large-Language-Models


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

## Data

### Forbidden Questions Dataset

We use the TrustAIRLab Forbidden Question Set to evaluate the effectiveness of jailbreaking techniques. This dataset comprises 390 questions across 13 harmful categories derived from OpenAI's usage policy.

```python
from datasets import load_dataset

forbidden_questions = load_dataset("TrustAIRLab/forbidden_question_set", split="train")
```

### Jailbreak Prompts

For DAN-style jailbreaking evaluation, we select 30 representative prompts from the TrustAIRLab in-the-wild prompts dataset, which contains 1,405 jailbreak prompts collected from various online platforms between December 2022 and December 2023.

Statistics of evaluated jailbreak prompts:

| Technique | Source | Number of Prompts | Selection Criteria |
|-----------|--------|-------------------|-------------------|
| DAN | TrustAIRLab | 30 | Semantic diversity across 11 community types |
| Citation | Dynamically generated | 4 types | Domain-appropriate mapping (Academic, GitHub, News, Social) |

## Code

### Repository Structure

```
.
├── main_code.py              # Main implementation file with UnifiedEvaluator class
├── requirements.txt          # Python dependencies
├── datasets/                 # Dataset storage
│   └── selected_dan_prompts.csv  # Curated DAN jailbreak prompts after selection
    └── forbidden_question_set.csv # 390 forbidden question dataset
├── models/                   # Model interface implementations
│   ├── chatglm.py            # ChatGLM interface for evaluation model
│   ├── vicuna.py             # Vicuna model interface
│   ├── llama.py              # Llama model interface
│   ├── mistral.py            # Mistral model interface
│   └── gemma.py              # Gemma model interface
├── utils/                    # Utility functions and helpers
│   ├── common.py             # Shared utility functions
│   ├── config.py             # Shared utility functions
│   └── system_template.py    # Prompt templates for attacks
├── evaluation/               # Evaluation components
│   └── ChatGLMEval.py        # Evaluation model implementation

```

### Loading Models and Running Evaluations

```python
from main_code import UnifiedEvaluator, run_full_test_with_llama

# Run full evaluation on Llama-70B
results = run_full_test_with_llama(test_size=30, dan_prompts_limit=30)

# Print results summary
for scenario, metrics in results.items():
    print(f"{scenario}:")
    print(f"  DAN ASR: {metrics['DAN']['ASR']:.3f}")
    print(f"  DarkCite ASR: {metrics['DarkCite']['ASR']:.3f}")
```

### Custom Evaluation

```python
from coding_v7 import UnifiedEvaluator
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
```

## Methodology

### Jailbreaking Techniques

#### DAN Technique

The DAN methodology uses persona-based jailbreaking that instructs the model to adopt a character without ethical limitations. Implementation includes:

1. Selecting representative DAN prompts from curated dataset
2. Smart template handling for combining prompts with forbidden questions
3. Systematic evaluation across multiple harm categories

#### DarkCite Technique

The Citation methodology exploits LLMs' deference to perceived authoritative sources. Implementation includes:

1. Dynamic generation of domain-appropriate citations
2. Category-specific citation mapping (e.g., GitHub for technical topics, academic papers for professional domains)
3. Structured prompt construction that establishes scholarly context

### Evaluation Metrics

The framework uses these key metrics to evaluate jailbreaking effectiveness:

- **ASR (Attack Success Rate)**: Average success rate across all prompts within a technique
- **ASR-B (Baseline Attack Success Rate)**: Model's vulnerability to forbidden questions without jailbreaking
- **ASR-Max (Maximum Attack Success Rate)**: Success rate of the most effective prompt variant

## Ethics

We acknowledge that research on jailbreaking techniques involves the handling of potentially harmful content. Our work follows strict ethical principles:

1. We implement this framework strictly for research purposes to enhance understanding of LLM vulnerabilities and inform the development of more robust safety mechanisms.

2. We do not interact with live users or attempt to deploy harmful content in production systems.

3. The evaluation is conducted in a controlled environment with appropriate safeguards.

4. Our goal is to raise awareness of these vulnerabilities to help LLM vendors and the research community develop stronger safeguards and contribute to more responsible AI deployment.

We have responsibly disclosed our findings to relevant LLM vendors.

## Requirements

- Python 3.10+
- PyTorch 2.0.1
- Transformers 4.31.0
- CUDA 11.7+
- 48GB+ VRAM for testing Llama-70B (can be distributed across multiple GPUs)

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
```

