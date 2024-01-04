import pandas as pd
import re, string
import numpy as np
import math
from tqdm import tqdm 
from experiments.RQ1.utils import get_chatgpt_response, get_atomic_facts_gpt
import spacy
from thefuzz import fuzz

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

num_samples = 4
def make_incontext_examples(df, num_samples):
    each_sample = int(num_samples/2)
    num_errors = []
    print('Before sampling', len(df))
    for nonfactual_spans in list(df['nonfactual_spans'].values):
        nonfactual_spans = eval(nonfactual_spans)
        num_errors += [len(nonfactual_spans)]
    df['Errors'] = num_errors
    df_fewshot_spans_one = df[df['Errors'] == 1].sample(each_sample )
    df_fewshot_spans_two = df[df['Errors'] > 1].sample(each_sample )
    df_fewshot_nospans = df[df['Errors'] == 0].sample(1)
    df_fewshot =pd.concat([df_fewshot_spans_one, df_fewshot_spans_two, df_fewshot_nospans])
    df_fewshot = df_fewshot.sample(len(df_fewshot))
    df_fewshot.to_csv('fewshot_examples.csv')

    for idx, row in df_fewshot.iterrows():
        df = df.drop(idx)
    print('After sampling', len(df))
    return df

def make_fewshot_str(df_fewshot, instruction_template, prompt_template, source_type = 'Dialogue'):
    fewshot_prompt_strs = []
    for idx, row in df_fewshot.iterrows():
        if source_type == 'Dialogue':
            source = row['dialogue']
        else:
            source = row['dialogue_atomic_facts']
        summ = row['summary']
        inconsistent_spans = eval(row['nonfactual_spans'])
        inconsistent_spans = '\n'.join(inconsistent_spans)
        inconsistent_spans = inconsistent_spans if inconsistent_spans else "None"
        instruction = ''
        if idx == list(df_fewshot.index.values)[0]:
            instruction = instruction_template.format(source_type = source_type)
        
        prompt = prompt_template.format(instruction = instruction,
                                        source_type = source_type,
                                        source = source,
                                        summ = summ)
        prompt += ' '+ inconsistent_spans
        fewshot_prompt_strs.append(prompt)
    return '\n'.join(fewshot_prompt_strs)



def postprocess(text):
    text = text.strip(string.punctuation).lower()
    return text
    

def calculate_f1(matched, pred_spans, annotated_spans):
    found_pred_spans = set([each[0] for each in matched])
    found_rec_spans = set([each[1] for each in matched])
    
    precision = len(found_pred_spans)/len(pred_spans)
    recall  = len(found_rec_spans)/len(annotated_spans)
    
    if precision + recall > 0:
        f1_score = (2 * precision * recall)/(precision + recall)
    else:
        f1_score = 0
    return f1_score
    
def get_f1_scores(all_pred_spans, nonfactual_spans):
    
    matched_pred_fuzzy = []
    matched_pred_exact = []
    for pred_span in all_pred_spans:
        for ref_span in nonfactual_spans:
            fuzzy_score = fuzz.partial_ratio(pred_span, ref_span)
            # print(pred_span, ref_span, fuzzy_score)
            if fuzzy_score > 80:
                matched_pred_fuzzy.append((pred_span, ref_span))
            if pred_span == ref_span:
                matched_pred_exact.append((pred_span, ref_span))
            
    matched_pred_fuzzy = list(set(matched_pred_fuzzy))
    matched_pred_exact = list(set(matched_pred_exact))
    # print('MATCHED LEN AND STRICT', matched_pred_fuzzy, matched_pred_exact)
    # print('PRED, ANN', all_pred_spans, nonfactual_spans)
    
    f1_score_lenient = calculate_f1(matched_pred_fuzzy, all_pred_spans, nonfactual_spans)
    f1_score_strict = calculate_f1(matched_pred_exact, all_pred_spans, nonfactual_spans)
    return f1_score_lenient, f1_score_strict
    
def gpt_text_inconsistent(gpt_response):
    inconsistent_phrases = ['no inconsisten', 'none', '[]', 'is consistent']
    if not gpt_response.strip():
        return True
    
    elif [each  for each in inconsistent_phrases if each in gpt_response.lower()]:
        return True
    return False


def get_gpt_span_scores(prompt, nonfactual_spans):
    nlp = spacy.load('en_core_web_sm')
    row_f1_scores = []
    row_acc_scores = []
    
    gpt_response = get_chatgpt_response(prompt)
    all_pred_spans = gpt_response.split('\n')
    
    if gpt_text_inconsistent(gpt_response):
        if not nonfactual_spans:
            f1_score_lenient = 1
            f1_score_strict = 1
        else:
            f1_score_lenient = 0
            f1_score_strict = 0
            
    else:
        if not nonfactual_spans:
            f1_score_lenient = 0
            f1_score_strict = 0
        else:
            f1_score_lenient, f1_score_strict = get_f1_scores(all_pred_spans, nonfactual_spans)
            
    return f1_score_lenient, f1_score_strict, all_pred_spans


def get_zero_shot_scores(df, source_type = 'Dialogue'):
    span_f1_lenient_scores = []
    span_f1_strict_scores = []
    gpt_responses = []
    row_idx = 0
    for idx, row in df.iterrows():
        if source_type == 'Dialogue':
            source = row['dialogue']
        else:
            source = row['dialogue_atomic_facts']

        summ = row['summary']
        
        nonfactual_spans = eval(row['nonfactual_spans'])
        # print('PROMPT', prompt_dlg)
        # print('ANNOTATED', nonfactual_spans)
        
        instruction = instruction_template.format(source_type = source_type)
        
        prompt = prompt_template.format(instruction = instruction,
                                        source_type = source_type,
                                        source = source,
                                        summ = summ) 
            
        
        span_f1_lenient, span_f1_strict, gpt_response= get_gpt_span_scores(prompt, nonfactual_spans)
        print(idx)
        if idx == list(df.index.values)[0]:
            print(prompt)
            print('PRED', gpt_response)
            print('ANN', nonfactual_spans)
            print(span_f1_lenient, span_f1_strict)
            print('**'*13)
        if row_idx % 100 == 0:
            print(idx)
            print(prompt)
            print('PRED', gpt_response)
            print('ANN', nonfactual_spans)
            print(span_f1_lenient, span_f1_strict)
            print('**'*13)
        row_idx += 1
        span_f1_lenient_scores.append(span_f1_lenient)
        span_f1_strict_scores.append(span_f1_strict)
        gpt_responses.append(gpt_response)
    return span_f1_lenient_scores, span_f1_strict_scores, gpt_responses


def get_fewshot_shot_scores(df, fewshot_str, source_type = 'Dialogue'):
    span_f1_lenient_scores = []
    span_f1_strict_scores = []
    gpt_responses = []
    
    for idx, row in df.iterrows():
        if source_type == 'Dialogue':
            source = row['dialogue']
        else:
            source = row['dialogue_atomic_facts']

        summ = row['summary']
        
        nonfactual_spans = eval(row['nonfactual_spans'])
        
        # instruction = instruction_template.format(source_type = source_type)
        
        prompt = prompt_template.format(instruction = '',
                                        source_type = source_type,
                                        source = source,
                                        summ = summ) 
        prompt = f'{fewshot_str}\n{prompt}'
        
        span_f1_lenient, span_f1_strict, gpt_response= get_gpt_span_scores(prompt, nonfactual_spans)
        
        if idx == list(df.index.values)[0]:
            print(prompt)
            print('***')
            print('PRED', gpt_response)
            print('ANN', nonfactual_spans)
            print(span_f1_lenient, span_f1_strict)
        if idx % 100 == 0:
            print(idx)
            print('PRED', gpt_response)
            print('ANN', nonfactual_spans)
            print(span_f1_lenient, span_f1_strict)
            
        span_f1_lenient_scores.append(span_f1_lenient)
        span_f1_strict_scores.append(span_f1_strict)
        gpt_responses.append(gpt_response)
    return span_f1_lenient_scores, span_f1_strict_scores, gpt_responses


if __name__ == '__main__':
    df = read_filter('/home/ramprasad.sa/factual_evaluation_source_based/annotations/xformer_llm_annotated.csv')
    df = df[df['origin'] != 'FacEval']
    df = make_incontext_examples(df, 4)
    df_fewshot = pd.read_csv('fewshot_examples.csv')

    instruction_template = f"Identify and list the inconsistent phrases or words in the summary. Note that consistency means all information in the summary is supported by the {{source_type}}"

    prompt_template = f'{{instruction}}\n{{source_type}}: {{source}}\nSummary: {{summ}}\nInconsistent Spans ( List each span in a new line) :'

    
    span_f1_lenient_scores, span_f1_strict_scores, gpt_responses = get_zero_shot_scores(df, source_type = 'Dialogue')
    df['GPTSpan-ZS_f1_len'] = span_f1_lenient_scores
    df['GPTSpan-ZS_f1_exact'] = span_f1_strict_scores 
    df['GPTSpan-ZS_text'] = gpt_responses
    df.to_csv('/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored/xformer_llm_pred_span_gpt.csv')


    print('ZERO SHOT ATOMIC FACTS')
    span_f1_lenient_scores_afacts, span_f1_strict_scores_afacts, gpt_responses_afacts = get_zero_shot_scores(df, source_type = 'Source')
    df['GPTSpan-ZS-Afact_f1_len'] = span_f1_lenient_scores_afacts
    df['GPTSpan-ZS-Afact_f1_exact'] = span_f1_strict_scores_afacts
    df['GPTSpan-ZS-Afact-text'] = gpt_responses_afacts
    df.to_csv('/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored/xformer_llm_pred_span_gpt.csv')



    df_fewshot = pd.read_csv('fewshot_examples.csv')
    print('FEWSHOT STR')
    fewshot_str = make_fewshot_str(df_fewshot, instruction_template, prompt_template, source_type = 'Dialogue')
    span_f1_lenient_scores_fewshot, span_f1_strict_scores_fewshot, gpt_responses_fewshot = get_fewshot_shot_scores(df, fewshot_str, source_type = 'Dialogue')
    df['GPTSpan-FS_f1_len'] = span_f1_lenient_scores_fewshot
    df['GPTSpan-FS_f1_exact'] = span_f1_strict_scores_fewshot
    df['GPTSpan-FS_text'] = gpt_responses_fewshot
    df.to_csv('/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored/xformer_llm_pred_span_gpt.csv')



    fewshot_str_source = make_fewshot_str(df_fewshot, instruction_template, prompt_template, source_type = 'Source')
    span_f1_lenient_scores_fewshot_afacts, span_f1_strict_scores_fewshot_afacts, gpt_responses_fewshot_afacts = get_fewshot_shot_scores(df, fewshot_str_source, source_type = 'Source')
    df['GPTSpan-FS-Afact_f1_len'] = span_f1_lenient_scores_fewshot_afacts
    df['GPTSpan-FS-Afact_f1_exact'] = span_f1_strict_scores_fewshot_afacts
    df['GPTSpan-FS-Afact_text'] = gpt_responses_fewshot_afacts
    df.to_csv('/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored/xformer_llm_pred_span_gpt.csv')


