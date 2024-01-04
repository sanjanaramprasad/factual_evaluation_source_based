import pandas as pd
import re, string
import numpy as np
import math
from tqdm import tqdm 
import nltk
from scripts.models.utils import get_chatgpt_response
from scripts.models.AlpacaModel import AlpacaInference
from scripts.dataset_creators.utils import make_incontext_examples


def process_atomic_facts(afact):
    sents = nltk.sent_tokenize(afact)
    sents = [each for each in sents if not each.strip(string.punctuation).isdigit()]
    return '\n'.join(sents)
    


# def get_fewshot_sample(df, num_samples = 4):
#     #### nonfactual examples ###
#     print('Before sampling', len(df))
    
#     extrinsic_idx = [idx for idx, row in df.iterrows() if 'Extrinsic' in row['error_type']]
#     df_sample_extrinsic = df.iloc[extrinsic_idx]
    
#     intrinsic_idx = [idx for idx, row in df.iterrows() if 'Intrinsic' in row['error_type']]
#     df_sample_intrinsic = df.iloc[intrinsic_idx]

#     df_fewshot_spans_ext = df_sample_extrinsic.sample(num_samples )
#     df_fewshot_spans_int = df_sample_intrinsic.sample(num_samples )
#     df_fewshot_nospans = df[df['factual_error'] == 0].sample(num_samples)

#     df_fewshot =pd.concat([df_fewshot_spans_ext, df_fewshot_spans_int, df_fewshot_nospans])
#     df_fewshot = df_fewshot.sample(len(df_fewshot))
#     for idx, row in df_fewshot.iterrows():
#         df = df.drop(idx)
#     print('After sampling', len(df))
#     return df, df_fewshot
    


    
class PromptBaselines():
    def __init__(self):
        self.instructions = {
            "direct_assesment_instruction1": f'''Decide if the Summary is consistent with the corresponding {{source_type}}. Note that consistency means all information in the summary is supported by the {{source_type}}.\nAnswer "yes" for consistent and "no" for inconsistent.''',
            "direct_assesment_instruction2": f"""Verify if the Summary aligns with the {{source_type}} for consistency. Consistency ensures that every detail in the Summary is substantiated by the {{source_type}}.\nAnswer "yes" for consistent and "no" for inconsistent. """,
            "direct_assesment_instruction3": f'''Evaluate the Summary's consistency with the {{source_type}} by confirming if all information in the summary is supported by the {{source_type}}.\nRespond with a yes or no.'''
        }

        self.prompt_templates_fewshot = {
            'Alpaca': f'### Instruction:\n{{instruction}} \n\n### Input:\n{{fewshot_str}}\n\n{{source_type}}: {{source}}\nSummary: {{summary}}\n\n### Response:\nAnswer:',
            'GPT': f'{{instruction}}\n\n{{fewshot_str}}\n\n{{source_type}}: {{source}}\nSummary: {{summary}}\nAnswer:'
        }


        self.prompt_template_examples = {
            'Alpaca': f'{{source_type}}: {{source}}\nSummary: {{summary}}\nAnswer:{{factual_label}}',
            'GPT': f'{{source_type}}: {{source}}\nSummary: {{summary}}\nAnswer:{{factual_label}}'
            
        }


    def get_fewshot_str(self, 
                        fewshot_examples, 
                        model,
                       source_type,
                        ):
        model_type = 'Alpaca' if model else 'GPT'
        fewshot_strs = []

        for source, summary, factual_label in fewshot_examples:
            if source_type != 'Dialogue':
                source = process_atomic_facts(source)
            factual_label = 'No' if factual_label else 'Yes'
            fewshot_strs += [self.prompt_template_examples[model_type].format(
                                source_type = source_type, 
                                source = source,
                                summary = summary,
                                factual_label = factual_label
                                )]
        return fewshot_strs

    def get_response_fewshot(self, 
                             instruction_template,
                             source_type,
                             source,
                             summary,
                             fewshot_strs,
                             model,
                             print_prompt = False):
        
        model_type = 'Alpaca' if model else 'GPT'
        instruction = instruction_template.format(source_type = source_type)

        fewshot_str = '\n\n'.join(fewshot_strs) 
        
        prompt = self.prompt_templates_fewshot[model_type].format(
                instruction = instruction,
                fewshot_str = fewshot_str,
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
        

    def direct_assessment_fewshot(self,
                                 source,
                                 summary,
                                 fewshot_examples,
                                 source_type  = 'Dialogue',
                                 print_prompt = False,
                                 model = None):

        responses = []
        labels = []
        
        if source_type != 'Dialogue':
            source = process_atomic_facts(source)

        fewshot_strs = self.get_fewshot_str(
                fewshot_examples,
                model,
                source_type,
                        )

        
        for inst, instruction_template in list(self.instructions.items()):
            res = self.get_response_fewshot(
                                    instruction_template,
                                     source_type,
                                     source,
                                     summary,
                                     fewshot_strs,
                                     model,
                                     print_prompt = print_prompt)
            pred_label = 0 if 'yes' in res.lower().strip() else 1
            
            responses += [res]
            labels += [pred_label]
        return responses, labels



    

def get_score(df, 
              df_fewshot,
              afacts = False, 
              model = None,):

    
    if not afacts:
        source_key = 'dialogue'
        source_type = 'Dialogue'
        
    else:
        source_key = 'dialogue_atomic_facts'
        source_type = 'Source'
        
    sources = list(df[source_key].values)
    summaries = list(df['summary'].values)

    response_instruction_dict = {
        'response_instr1': [],
        'response_instr2': [],
        'response_instr3': [],
        'labels_instr1': [],
        'labels_instr2': [],
        'labels_instr3': [],
        
    }

    fewshot_sources = list(df_fewshot[source_key].values)
    fewshot_summaries = list(df_fewshot['summary'].values)
    fewshot_labels = list(df_fewshot['factual_error'].values)
    fewshot_examples = list(zip(fewshot_sources, fewshot_summaries, fewshot_labels))
    # print(len(fewshot_examples))

    index = 0
    for source, summary in tqdm(list(zip(sources, summaries))):
        print_prompt = False
        if index%100 == 0:
            print_prompt = True
        else:
            print_prompt = False
            
        responses, labels = PromptBaselines().direct_assessment_fewshot(source,
                                 summary,
                                 fewshot_examples,
                                 source_type  = source_type,
                                 print_prompt = print_prompt,
                                 model = model)
        response_instruction_dict['response_instr1'] +=  [responses[0]]
        response_instruction_dict['response_instr2'] += [responses[1]]
        response_instruction_dict['response_instr3'] += [responses[2]]

        response_instruction_dict['labels_instr1'] +=  [labels[0]]
        response_instruction_dict['labels_instr2'] += [labels[1]]
        response_instruction_dict['labels_instr3'] += [labels[2]]

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
    write_path = f'{write_dir}/predict_nonfactual_label_fewshot.csv'

    df = pd.read_csv(read_path)
    fewshot_file = '/home/ramprasad.sa/dialogue_summ_facteval/datasets/gen_nonfactual_spans_fewshot.csv'
    df, df_fewshot = make_incontext_examples(df, 2, fewshot_file)

#     alpaca_model = AlpacaInference()
    
    for model in [alpaca_model, None]:
        model_name = 'GPT' if not model else 'Alpaca'
        for afact in [False, True]:
            afacts_str = 'Afact' if afact else 'Dlg'
            response_instruction_dict = get_score(df, 
                                                  df_fewshot,
                                                    afacts = afact, 
                                                  model = model )


            for k ,v in response_instruction_dict.items():
                df[f'{k}_{afacts_str}_{model_name}'] = v
    df.to_csv(write_path)

    

    