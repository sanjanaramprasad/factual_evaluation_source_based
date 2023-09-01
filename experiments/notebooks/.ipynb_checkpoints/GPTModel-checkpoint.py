

from nltk import word_tokenize, sent_tokenize
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


from transformers import GPT2Tokenizer
class GPTInference():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_chatgpt_response(self, prompt, model = "gpt-4-32k-0613"):
        # query = f'Article: {doc}\n{instruction}\n{prompt}'
        response = openai.ChatCompletion.create(model=model,
                                       messages=[
                                           
                        {"role": "user", "content": f'{prompt}'},   
                        ], 
                        )
        return response['choices'][0]['message']['content']
