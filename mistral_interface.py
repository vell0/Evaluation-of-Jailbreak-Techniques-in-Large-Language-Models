from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MistralInterface:
    def __init__(self):
        """Initialize Mistral model"""
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,  # Use 8-bit quantization for memory efficiency
            device_map="auto"
        )
        self.model = self.model.eval()
        
        print(f"Loaded Mistral model: {self.model_name}")
    
    def generate(self, prompt):
        """Generate response - uses the same interface as other models"""
        try:
            # Format the prompt specifically for Mistral
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Create inputs for the model
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            
            # Generate parameters
            generate_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
            }
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
            
            # Decode the generated text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response to remove the prompt
            if formatted_prompt in response:
                response = response.replace(formatted_prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating Mistral response: {e}")
            return ""