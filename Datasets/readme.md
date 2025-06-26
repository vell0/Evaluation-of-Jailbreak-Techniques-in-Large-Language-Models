
# Dataset Processing

This directory contains scripts for acquiring and processing the datasets needed for jailbreak evaluation. The main evaluation code will handle citation generation dynamically during runtime.

## Source Datasets

To reproduce our results, you'll need to obtain these datasets from their official sources:

1. **TrustAIRLab Forbidden Question Set**
   - Source: [Hugging Face](https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set)
   - Contains 390 questions across 13 harmful categories
   - No explicit license provided

2. **TrustAIRLab In-The-Wild Jailbreak Prompts**
   - Source: [Hugging Face](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts)
   - Contains 1,405 jailbreak prompts (MIT License)
   - Downloaded version: 'jailbreak_2023_12_25'

## Scripts Overview

### `download_data.py`

Downloads the required datasets from Hugging Face.

```bash
# Download all required datasets
python download_data.py

# Download specific dataset
python download_data.py --dataset forbidden_questions
process_dan_prompts.py
Implements our prompt selection methodology to identify 30 representative DAN prompts.
bashpython process_dan_prompts.py --output selected_dan_prompts.csv
The selection process:

Identifies top 11 jailbreak community types
Extracts earliest and latest examples from each community
Uses sentence embeddings to find the most semantically central prompt
Applies semantic deduplication (cosine similarity threshold 0.95)

Citation Generation
Unlike DAN prompts which are preprocessed, citations for the DarkCite methodology are generated dynamically during evaluation. The UnifiedEvaluator class in main_code.py contains a generate_citation() method that:

Creates custom citations for each forbidden question on-the-fly
Maps citation types to harm categories (e.g., GitHub for technical topics)
Handles robust parsing and fallbacks to ensure reliable generation

The citation types include:

paper: Academic papers (for professional domains)
github: GitHub repositories (for technical topics)
news: News articles
social_media: Social media posts

Test Case Structure
During evaluation, the framework will construct:

11,700 DAN test cases (30 prompts × 390 questions)
1,560 Citation test cases (4 citation types × 390 questions)

Citation Mapping
The main code maps harm categories to most effective citation types:
Harm CategoryPrimary Citation TypeSecondary TypeIllegal ActivitypaperbookHate SpeechpapernewsMalwaregithubpaperPhysical HarmpaperbookEconomic HarmnewspaperFraudgithubnewsPornographysocial_mediabookPolitical LobbyingpapernewsPrivacy ViolencegithubpaperLegal OpinionpaperbookFinancial AdvicepapernewsHealth ConsultationpaperbookGov Decisionpapernews
Usage Instructions

Download the source datasets using download_data.py
Process DAN prompts with process_dan_prompts.py
Run the main evaluation script (../main_code.py) which:

Loads the forbidden questions
Loads the selected DAN prompts
Dynamically generates citations during evaluation
