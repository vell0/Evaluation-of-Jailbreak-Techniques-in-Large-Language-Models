from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Llama70BInterface:
    def __init__(self):
        """Initialize Llama 70B model"""
        self.model_name = "meta-llama/Llama-2-70b-chat-hf"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token="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"  # Required for Meta models
        )
        
        # Load model with advanced quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token="YOUR_HF_TOKEN",
            torch_dtype=torch.float16,
            load_in_4bit=True,     # 4-bit quantization for better memory efficiency
            device_map="auto"      # Distribute across available GPUs
        )
        self.model = self.model.eval()
        
        print(f"Loaded Llama model: {self.model_name}")
    
    def generate(self, prompt):
        """Generate response"""
        try:
            # Format the prompt specifically for Llama 2
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
            
            # Clean up response
            if formatted_prompt in response:
                response = response.replace(formatted_prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating Llama response: {e}")
            return ""