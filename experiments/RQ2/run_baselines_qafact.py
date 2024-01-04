import pandas as pd
from qafacteval import QAFactEval


kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
        "verbose": True, "generation_batch_size": 32, \
        "answering_batch_size": 32, "lerc_batch_size": 8}

model_filter = {
    'FacEval': ['bart_large', 'co-ref bart large', 'condigsum bart large','gpt4-32k-0613','mv-bart_large', 'alpaca-13b'],
    'SAMSum': ['BART', 'CODS', 'MV-BART', 'UniLM', 'gpt4-32k-0613', 'alpaca-13b'],
    'DialogueSum': ['BART', 'CODS', 'MV-BART', 'UniLM', 'gpt4-32k-0613', 'alpaca-13b']

}

model_folder = "/home/ramprasad.sa/factual_evaluation_source_based/experiments/RQ2/QAFactEval/models" # path to models 
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)
      


### Use environment QAFact


def run_qafacteval(sources, summaries):
    results = []
    for src, summ in list(zip(sources, summaries)):
        # print(src, summ)
        res = metric.score_batch_qafacteval([src], [[summ]], return_qa_pairs=True)
        results.append(res[0][0]['qa-eval']['lerc_quip'])
    return results



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


read_dir =  '/home/ramprasad.sa/factual_evaluation_source_based/annotations'
read_path = f'{read_dir}/xformer_llm_annotated.csv'
    
write_dir = '/home/ramprasad.sa/factual_evaluation_source_based/datasets/scored'
write_path = f'{write_dir}/baseline_metrics_qafact.csv'
df = read_filter(read_path)
dialogues = list(df['dialogue'].values)
sources_atomic_facts = list(df['dialogue_atomic_facts'].values)
summaries = list(df['summary'].values)

qafacteval_scores = run_qafacteval(dialogues, summaries)
df['QAFactEval_scores'] = qafacteval_scores

qafacteval_scores_afacts = run_qafacteval(sources_atomic_facts, summaries)
df['QAFactEval-Afacts_scores'] = qafacteval_scores_afacts
df.to_csv(write_path)
