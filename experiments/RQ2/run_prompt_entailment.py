import pandas as pd
import re, string
import numpy as np
import math
from tqdm import tqdm 
import nltk
from experiments.RQ1.utils import get_chatgpt_response, get_atomic_facts_gpt
from experiments.models.AlpacaModel import AlpacaInference


def process_atomic_facts(afact):
    sents = nltk.sent_tokenize(afact)
    sents = [each for each in sents if not each.strip(string.punctuation).isdigit()]
    return '\n'.join(sents)
    
class PromptBaselines():
    def __init__(self):
        self.instructions = {
            "direct_assesment_instruction1": f'''Decide if the Summary entails information in the corresponding {{source_type}}.\nAnswer "yes" or "no".''',
        }

        self.prompt_templates_zeroshot = {
            'GPT': f'{{instruction}}\n{{source_type}}: {{source}}\nSummary: {{summary}}\nAnswer:'
        }


    def get_response_zeroshot(self,
                                     instruction_template,
                                   source_type,
                                   source,
                                   summary,
                                    model,
                                   print_prompt = False):

        model_type = 'Alpaca' if model else 'GPT'
        instruction = instruction_template.format(source_type = source_type)
            
        prompt = self.prompt_templates_zeroshot[model_type].format(
                instruction = instruction,
                source_type = source_type,
                source = source,
                summary = summary
            )
        
        if print_prompt:
            print('PROMPT', prompt)
            print('***')
    

        if not model:
                response = get_chatgpt_response(prompt, 'gpt-4-0613')
        else:
                response = model.get_response(prompt, max_len = None)
    
        
        return response

    def direct_assessment_zeroshot(self, 
                         source, 
                         summary,
                         source_type  = 'Dialogue',
                         print_prompt = False,
                         model = None):

        responses = []
        labels = []
        
        if source_type != 'Dialogue':
            source = process_atomic_facts(source)

        

        
        for inst, instruction_template in self.instructions.items():
            res = self.get_response_zeroshot(
                                    instruction_template,
                                   source_type,
                                   source,
                                   summary,
                                    model,
                                   print_prompt )
            pred_label = 0 if 'yes' in res.lower().strip() else 1
            
            responses += [res]
            labels += [pred_label]
        return responses, labels


def get_score(df, 
              eval_type, 
              afacts = False, 
              model = None):

    
    if not afacts:
        source_key = 'dialogue'
        source_type = 'Dialogue'
        
    else:
        source_key = 'dialogue_atomic_facts'
        source_type = 'Source'
        
    sources = list(df[source_key].values)
    summaries = list(df['summary'].values)

    response_instruction_dict = {
        'response_ent1': [],
        'labels_ent1': [],
        
    }
    
    
    index = 0
    for src, summ in tqdm(list(zip(sources, summaries))):
        print_prompt = False
        if index%100 == 0:
            print_prompt = True
        else:
            print_prompt = False
        if eval_type == 'direct_assesment':
            responses, labels = PromptBaselines().direct_assessment_zeroshot(src, 
                                                            summ, 
                                                            source_type  = source_type, 
                                                            model = model,
                                                            print_prompt = print_prompt,
                                                            )

            response_instruction_dict['response_ent1'] +=  [responses[0]]
            # response_instruction_dict['response_instr2'] += [responses[1]]
            # response_instruction_dict['response_instr3'] += [responses[2]]

            response_instruction_dict['labels_ent1'] +=  [labels[0]]
            # response_instruction_dict['labels_instr2'] += [labels[1]]
            # response_instruction_dict['labels_instr3'] += [labels[2]]

            
        
        
        index += 1
        if print_prompt:
            print('******')
            print(labels)
            print('RESPONSE', responses ,' ---')
            print('*****')
            
    return response_instruction_dict
        

if __name__ == '__main__':
    read_dir =  '/home/ramprasad.sa/factual_evaluation_source_based/annotations'
    read_path = f'{read_dir}/xformer_llm_annotated.csv'
    
    write_dir = '/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored'
    write_path = f'{write_dir}/predict_entailment_label_zeroshot.csv'

    
    df = pd.read_csv(read_path)
    # alpaca_model = AlpacaInference()
    
    for model in [None]:
        model_name = 'GPT' if not model else 'Alpaca'
        print('===' * 8, model_name, '===' * 8)
        for afact in [True, False]:
            afacts_str = 'Afact' if afact else 'Dlg'
            
            print('===' * 5, afacts_str, '===' * 5)
            response_instruction_dict_da_zs = get_score(df, 
                                                        eval_type = 'direct_assesment',
                                                        afacts = afact,
                                                        model = model)

            

            for k ,v in response_instruction_dict_da_zs.items():
                df[f'{k}_{afacts_str}_{model_name}_zeroshot'] = v


    df.to_csv(write_path)


    

    