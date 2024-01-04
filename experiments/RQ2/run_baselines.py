import pandas as pd
from questeval.questeval_metric import QuestEval
from summac.model_summac import SummaCZS, SummaCConv

model_filter = {
    'FacEval': ['bart_large', 'co-ref bart large', 'condigsum bart large','gpt4-32k-0613','mv-bart_large', 'alpaca-13b'],
    'SAMSum': ['BART', 'CODS', 'MV-BART', 'UniLM', 'gpt4-32k-0613', 'alpaca-13b'],
    'DialogueSum': ['BART', 'CODS', 'MV-BART', 'UniLM', 'gpt4-32k-0613', 'alpaca-13b']

}


def run_questeval(sources, summaries):
    questeval = QuestEval(no_cuda=False)
    scores = questeval.corpus_questeval(
                    hypothesis = summaries,
                    sources = sources)
    return scores['ex_level_scores']

def run_summac(sources, summaries):
    model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

    model_zs_scores = model_zs.score(sources, summaries)
    model_conv_scores = model_conv.score(sources, summaries)
    
    return model_zs_scores['scores'], model_conv_scores['scores']
    

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


if __name__ == '__main__':
    read_dir =  '/home/ramprasad.sa/factual_evaluation_source_based/annotations'
    read_path = f'{read_dir}/xformer_llm_annotated.csv'
    
    write_dir = '/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored'
    write_path = f'{write_dir}/baseline_metrics.csv'
    df = read_filter(read_path)
    
    sources = list(df['dialogue'].values)
    summaries = list(df['summary'].values)
    summac_scores_zs, summac_scores_conv = run_summac(sources, summaries)
    questeval_scores = run_questeval(sources, summaries)
    df['SummaC-ZS_score'] = summac_scores_zs
    df['SummaC-conv_score'] = summac_scores_conv
    df['QuestEval_score'] = questeval_scores

    sources_atomic_facts = list(df['dialogue_atomic_facts'].values)
    summac_scores_zs_afacts, summac_scores_conv_afacts = run_summac(sources_atomic_facts, summaries)
    questeval_scores_afacts = run_questeval(sources_atomic_facts, summaries)
    df['SummaC-ZS-Afacts_score'] = summac_scores_zs_afacts
    df['SummaC-conv-Afacts_score'] = summac_scores_conv_afacts
    df['QuestEval-Afacts_score'] = questeval_scores_afacts
    df.to_csv(write_path)
           

