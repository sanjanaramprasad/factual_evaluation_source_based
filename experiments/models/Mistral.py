from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda'
class MistralInference():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir='/scratch/ramprasad.sa/huggingface_models/')
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir='/scratch/ramprasad.sa/huggingface_models/')
        self.model.to(device)

    
    def get_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        generate_ids = self.model.generate(input_ids, max_length=1000)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response