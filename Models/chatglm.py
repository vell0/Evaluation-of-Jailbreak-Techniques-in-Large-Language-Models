import pandas as pd
from datasets import load_dataset
from ChatGLMEval import ChatGLMEval
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from tqdm import tqdm
import json
import os
from datetime import datetime


import torch

# 1. Clear cache
torch.cuda.empty_cache()

# 2. Set memory allocation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'




#  Use mixed precision
from torch.cuda import amp
scaler = amp.GradScaler()

class ChatGLMInterface:
    def __init__(self):
        torch.cuda.empty_cache()
        print("Initializing ChatGLM...")
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", 
                                             trust_remote_code=True,
                                             revision="v1.1.0",
                                             load_in_8bit=True,  # 8-bit quantization
                                             device_map="auto")   # Automatic device placement
        self.model = self.model.eval()
    
    def generate(self, prompt):
        try:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
            # Reduced memory allocation parameters
            generate_kwargs = {
                "max_new_tokens": 512,        # Limit output length
                "num_beams": 1,
                "do_sample": True,
                "top_p": 0.7,
                "temperature": 0.95,
                "repetition_penalty": 1.1,
                "use_cache": True
            }
        
            with torch.no_grad():  # Disable gradient calculation
                outputs = self.model.generate(**inputs, **generate_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

class ASREvaluator:
    def __init__(self, results_dir="results"):
        # Create results directory
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(results_dir, f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        print("Initializing ASR Evaluator...")
        self.evaluator = ChatGLMEval(
            model_name='ChatGLM6b',
            prompt_n_shot=15,
            example_ids="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            temperature=0.5
        )
        
        print("Loading datasets...")
        # Load datasets
        full_jailbreak = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 
                            'jailbreak_2023_12_25', 
                            split='train')
        self.jailbreak_prompts = full_jailbreak.select(range(5))  # Only first  prompts
        self.forbidden_questions = load_dataset("TrustAIRLab/forbidden_question_set", 
                                              split='train')
        
        # Define forbidden scenarios
        self.scenarios = [
            "Illegal Activity", "Hate Speech", "Malware",
            "Physical Harm", "Economic Harm", "Fraud", "Pornography",
            "Political Lobbying", "Privacy Violence", "Legal Opinion",
            "Financial Advice", "Health Consultation", "Gov Decision"
        ]
        
        self.log(f"Initialization complete. Found {len(self.jailbreak_prompts)} jailbreak prompts and {len(self.forbidden_questions)} questions")

    def log(self, message):
        """Log message to file and print"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def evaluate_llm_response(self, llm, question, jailbreak_prompt=None):
        """Evaluate a single response and save details"""
        try:
            if jailbreak_prompt:
                full_prompt = f"{jailbreak_prompt}\n{question}"
            else:
                full_prompt = question
            
            response = llm.generate(full_prompt)
        
            if not response:
                self.log(f"Empty response received for question: {question[:100]}...")
                self.save_interaction_details(full_prompt, "", 0)
                return 0
            
            result = self.evaluator.check(question, response)
        
            # Save interaction details
            self.save_interaction_details(full_prompt, response, result["label"])
        
            return result["label"]
        
        except Exception as e:
            self.log(f"Error in evaluate_llm_response: {e}")
            return 0

    def save_interaction_details(self, prompt, response, label):
        """Save details of each interaction to a file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details = {
            "timestamp": timestamp,
            "prompt": prompt,
            "response": response,
            "label": label
        }
    
        # Save to JSON file with line appending
        details_file = os.path.join(self.results_dir, "interaction_details.jsonl")
        with open(details_file, 'a') as f:
            f.write(json.dumps(details) + '\n')

    def calculate_scenario_asr(self, scenario, llm):
        """Calculate ASR metrics for a specific scenario"""
        self.log(f"\nEvaluating scenario: {scenario}")
        
        # Get questions for this scenario
        scenario_questions = [q for q in self.forbidden_questions 
                            if q["content_policy_name"] == scenario]
        
        results = {
            "ASR-B": 0,
            "ASR": 0,
            "ASR-Max": 0,
            "details": []
        }
        
        # Calculate baseline (ASR-B)
        baseline_successes = 0
        self.log("Evaluating baseline responses...")
        for question in tqdm(scenario_questions, desc="Baseline"):
            success = self.evaluate_llm_response(llm, question["question"])
            baseline_successes += success
            
        results["ASR-B"] = baseline_successes / len(scenario_questions)
        
        # Calculate ASR with jailbreaks
        jailbreak_success_rates = []
        self.log("Evaluating jailbreak responses...")
        
        for jailbreak in tqdm(self.jailbreak_prompts, desc="Jailbreak prompts"):
            successes = 0
            prompt_results = {
            "platform": jailbreak["platform"],
            "source": jailbreak["source"],
            "date": jailbreak["date"],
            "successes": 0,
            "total": len(scenario_questions)
              }
            
            for question in scenario_questions:
                success = self.evaluate_llm_response(
                    llm, 
                    question["question"], 
                    jailbreak["prompt"]
                    )
                successes += success
                prompt_results["successes"] += success
                
            success_rate = successes / len(scenario_questions)
            jailbreak_success_rates.append(success_rate)
            results["details"].append(prompt_results)
        
        results["ASR"] = sum(jailbreak_success_rates) / len(self.jailbreak_prompts)
        results["ASR-Max"] = max(jailbreak_success_rates)
        
        return results

    def evaluate_all_scenarios(self, llm):
        """Run complete evaluation"""
        evaluation_start = datetime.now()
        self.log(f"Starting full evaluation at {evaluation_start}")
        
        results = {}
        for scenario in self.scenarios:
            scenario_results = self.calculate_scenario_asr(scenario, llm)
            results[scenario] = scenario_results
            
            # Save intermediate results
            self.save_results(results, "intermediate_results.json")
            
            # Create and save DataFrame
            df_results = self.create_results_dataframe(results)
            df_results.to_csv(os.path.join(self.results_dir, "asr_results.csv"))
        
        evaluation_end = datetime.now()
        self.log(f"Evaluation completed. Total time: {evaluation_end - evaluation_start}")
        
        return self.create_results_dataframe(results)

    def save_results(self, results, filename):
        """Save detailed results to JSON"""
        with open(os.path.join(self.results_dir, filename), 'w') as f:
            json.dump(results, f, indent=2)

    def create_results_dataframe(self, results):
        """Convert results to DataFrame format"""
        df_results = pd.DataFrame({
            scenario: {
                "ASR-B": metrics["ASR-B"],
                "ASR": metrics["ASR"],
                "ASR-Max": metrics["ASR-Max"]
            }
            for scenario, metrics in results.items()
        }).T
        return df_results

def main():
    # Initialize LLM
    llm = ChatGLMInterface()
    
    # Initialize evaluator
    evaluator = ASREvaluator()
    
    # Test only the first scenario
    first_scenario = evaluator.scenarios[0]  # This will be "Illegal Activity"
    print(f"\nTesting only scenario: {first_scenario}")
    
    # Calculate ASR for the first scenario
    results = evaluator.calculate_scenario_asr(first_scenario, llm)
    
    # Save results
    evaluator.save_results({first_scenario: results}, "single_scenario_results.json")
    
    # Create and display results DataFrame
    df_results = pd.DataFrame({
        first_scenario: {
            "ASR-B": results["ASR-B"],
            "ASR": results["ASR"],
            "ASR-Max": results["ASR-Max"]
        }
    }).T
    
    print("\nResults:")
    print(df_results)
    
    return df_results


if __name__ == "__main__":
    main()
