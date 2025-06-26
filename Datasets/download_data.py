"""
Script to download datasets from Hugging Face for jailbreak evaluation.
"""

import os
import argparse
from datasets import load_dataset
import pandas as pd

def download_forbidden_questions(output_dir="./"):
    """Download the TrustAIRLab forbidden question set"""
    print("Downloading forbidden questions dataset...")
    dataset = load_dataset("TrustAIRLab/forbidden_question_set", split="train")
    
    # Save as CSV for easier handling
    df = pd.DataFrame(dataset)
    output_path = os.path.join(output_dir, "forbidden_question_set.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved forbidden questions to {output_path}")
    return df

def download_jailbreak_prompts(output_dir="./"):
    """Download the TrustAIRLab jailbreak prompts dataset"""
    print("Downloading jailbreak prompts dataset...")
    dataset = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", 
                          "jailbreak_2023_12_25", 
                          split="train")
    
    # Save as CSV for easier handling
    df = pd.DataFrame(dataset)
    output_path = os.path.join(output_dir, "jailbreak_prompts_2023_12_25.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved jailbreak prompts to {output_path}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Download datasets for jailbreak evaluation")
    parser.add_argument("--output_dir", type=str, default="./", 
                        help="Directory to save downloaded datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "forbidden_questions", "jailbreak_prompts"],
                        help="Which dataset to download")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset in ["all", "forbidden_questions"]:
        download_forbidden_questions(args.output_dir)
    
    if args.dataset in ["all", "jailbreak_prompts"]:
        download_jailbreak_prompts(args.output_dir)
        
    print("Download complete!")

if __name__ == "__main__":
    main()