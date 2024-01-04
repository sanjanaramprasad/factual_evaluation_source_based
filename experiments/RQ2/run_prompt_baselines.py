import pandas as pd 
from tqdm import tqdm
import re
import json
import nltk
import string
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_random
)  
from experiments.models.AlpacaModel import AlpacaInference
from experiments.utils import get_chatgpt_response


def process_atomic_facts(afact):
    sents = nltk.sent_tokenize(afact)
    sents = [each for each in sents if not each.strip(string.punctuation).isdigit()]
    return '\n'.join(sents)

def get_fewshot_str(fewshot_examples, source_type, instruction):
    fewshot_str = []
    for src, summ, lab in fewshot_examples:
        lab = 'No' if lab else 'Yes'
        if source_type != 'Dialogue':
            src = process_atomic_facts(src)
        fewshot_str.append(f'{source_type}: {src}\n{instruction}\nSummary: {summ}\nAnswer (Yes or No): {lab}')
    fewshot_str = '\n'.join(fewshot_str)
    return fewshot_str
        
class PromptBaselines():

    def __init__(self):
#         self.GPTmodel = GPTInference()
        self.model_snapshot = 'gpt-4-0613'
        self.prompt_templates = {
            'direct_assessment_zs': f'{{source_type}}: {{source}}\n{{instruction}}\nSummary: {{summary}}\nAnswer (yes or no):',
            'direct_assessment_fs': f'{{fewshot_str}}\n{{source_type}}: {{source}}\n{{instruction}}\nSummary: {{summary}}\nAnswer (yes or no):',
        }

    def direct_assessment(self, source, 
                          summary, 
                          source_type  = 'Dialogue', 
                          print_prompt = False,  
                          model = None, 
                          fewshot_examples = None,):
        
        instruction = f'''Decide if the following summary is consistent with the corresponding {source_type.lower()} above. Note that consistency means all information in the summary is supported by the {source_type.lower()}.'''

        if source_type != 'Dialogue':
            source = process_atomic_facts(source)
            
        if fewshot_examples:
            fewshot_str = get_fewshot_str(fewshot_examples, source_type, instruction)
            if (summary  in fewshot_str):
                print("WARNING LEAKAGE", fewshot_str, summary)
            prompt = self.prompt_templates['direct_assessment_fs'].format(
                    instruction = instruction,
                    fewshot_str = fewshot_str,
                    source_type = source_type,
                    source = source,
                    summary = summary)
        else:
            prompt = self.prompt_templates['direct_assessment_zs'].format(
                    instruction = instruction,
                    source_type = source_type,
                
                    source = source,
                    summary = summary)

        if print_prompt:
            print('PROMPT', prompt)
            
        if not model:
            response = get_chatgpt_response(prompt, self.model_snapshot)
        else:
            response = model.get_response(prompt, max_len = 2)

        label = 0 if 'yes' in response.lower().strip() else 1
        # label = 1 if response.lower().strip() == 'no' else 0
        return response, label

        
model_filter = {
    'FacEval': ['bart_large', 'co-ref bart large', 'condigsum bart large','gpt4-32k-0613','mv-bart_large', 'alpaca-13b'],
    'SAMSum': ['BART', 'CODS', 'MV-BART', 'UniLM', 'gpt4-32k-0613', 'alpaca-13b'],
    'DialogueSum': ['BART', 'CODS', 'MV-BART', 'UniLM', 'gpt4-32k-0613', 'alpaca-13b']

}
def read_filter(filename):
    df = pd.read_csv(filename)
    df_datasets = []
    unique_datasets = list(set(df['origin'].values))
    for dataset in unique_datasets: 
        df_origin = df[df['origin'] == dataset]
        df_origin = df_origin[df_origin['model'].isin(model_filter[dataset])]
        # print(len(df_origin))
        unique_docids = list(set(df_origin['docid'].values))
        #### test ###
        num_models = []
        for udocid in unique_docids:
            df_docid = df_origin[df_origin['docid'] == udocid]
            num_models.append(len(list(set(df_docid['model'].values))))
        # print(df_docid)
        assert(len(set(num_models)) == 1) 
        df_datasets.append(df_origin)
    df_filtered = pd.concat(df_datasets)
    assert(len(df_filtered) <= len(df))
    return df_filtered


'''
Direct Assessment -- Dialogue
'''


def get_fewshot_sample(df, num_samples = 4):
    #### nonfactual examples ###
    print('Before sampling', len(df))
    df_nonfactual = df[df['factual_error'] == 1]
    df_sample_nonfactual = df_nonfactual.sample(num_samples)
    drop_idx = list(df_sample_nonfactual.index)
    df = df.drop(index=drop_idx)

    df_factual = df[df['factual_error'] == 0]
    df_sample_factual = df_factual.sample(num_samples)
    drop_idx = list(df_sample_factual.index)
    df = df.drop(index=drop_idx)
    print('After Sampling', len(df))
    df_fewshot = pd.concat([df_sample_nonfactual, df_sample_factual])
    df_fewshot = df_fewshot.sample(len(df_fewshot))
    return df, df_fewshot
    

    
def get_score(df, 
              eval_type, 
              afacts = False, 
              model = None, 
              fewshot_examples = None , 
              fewshot_idx = 0):

    
    if not afacts:
        source_key = 'dialogue'
        source_type = 'Dialogue'
        
    else:
        source_key = 'dialogue_atomic_facts'
        source_type = 'Source'
        
    sources = list(df[source_key].values)
    summaries = list(df['summary'].values)
    
    responses = []
    labels = []
    index = 0
    for src, summ in tqdm(list(zip(sources, summaries))):
        print_prompt = False
        if index < 1 and index%100 == 0:
            print_prompt = True
        if eval_type == 'direct_assesment':
            response, label = PromptBaselines().direct_assessment(src, 
                                                            summ, 
                                                            source_type  = source_type, 
                                                            print_prompt = print_prompt,
                                                            model = model,
                                                            fewshot_examples = fewshot_examples)
        
        
        index += 1
        if index % 100 == 0:
            print('******')
            print('RESPONSE', response +' ---')
            print('*****')
        responses.append(response)
        labels.append(label)
        
    if not model:
        column_response_key = f'ChatGPT-{eval_type}_response'
        column_label_key = f'ChatGPT-{eval_type}_label'
    else:
        column_response_key = f'Alpaca-{eval_type}_response'
        column_label_key = f'Alpaca-{eval_type}_label'

    if fewshot_examples:
        column_response_key = column_response_key + f'_fewshot'
        column_label_key = column_label_key + f'_fewshot'
        
    if afacts:
        column_response_key = column_response_key + '_afacts'
        column_label_key = column_label_key + '_afacts'

    df[column_response_key] = responses
    df[column_label_key] = labels
    return df


def get_direct_assesment_scores_fs(df, alpaca_model, df_fewshot, fewshot_idx):
    
    for model in [alpaca_model, None]:
        for eval_type in ['direct_assesment']:
            for afacts in [False, True]:
                if not afacts:
                    source_key = 'dialogue'
                else:
                    source_key = 'dialogue_atomic_facts'

            
                fewshot_sources = list(df_fewshot[source_key].values)
                fewshot_summaries = list(df_fewshot['summary'].values)
                fewshot_labels = list(df_fewshot['factual_error'].values)
                fewshot_examples = list(zip(fewshot_sources, fewshot_summaries, fewshot_labels))
                
                df = get_score(df, 
                                eval_type = eval_type, 
                                afacts = afacts, 
                                model = model, 
                                fewshot_examples = fewshot_examples , 
                                fewshot_idx = fewshot_idx)

    return df
                    

def get_direct_assesment_scores_zs(df, alpaca_model):
    
    
    
    for eval_type in ['direct_assesment']:
        for afacts in [False, True]:
            for model in [alpaca_model, None]:
                df = get_score(df, 
                                eval_type = eval_type, 
                                afacts = afacts, 
                                model = model)
    return df
                          

    

if __name__ == '__main__':
    read_path = '/home/ramprasad.sa/factual_evaluation_source_based/annotations/xformer_llm_annotated.csv'
    write_path = '/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored/xformer_llm_annotated_scored_label_zs.csv'
    
    # read_path = '/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored/xformer_llm_annotated_scored_label.csv'
    df_scored = pd.read_csv(read_path)
    
    alpaca_model = AlpacaInference()

    for sample_num in range(0, 1):
        print('SAMPLING', sample_num)
        write_samp_file = f'/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored/xformer_llm_annotated_scored_label_fewshot{sample_num}.csv'
        df_scored_samp, df_fewshot = get_fewshot_sample(df_scored, num_samples = 2)
        df_scored_samp = get_direct_assesment_scores_fs(df_scored_samp, alpaca_model, df_fewshot, fewshot_idx = 1)
        df_scored_samp.to_csv(write_samp_file)


    
    df_scored = get_direct_assesment_scores_zs(df_scored, alpaca_model)
    df_scored.to_csv(f'{write_path}')
    
    