import pandas as pd

df = pd.read_csv('results/eval_results.csv')
focused_tasks = [
    'answer_verification',
    'irony_detection',
    'toxic_language_detection',
    'word_semantics'
]
# find rows where task != finetuned_on

cross_task_df = df[df['task'] != df['finetuned_on']]
# ignore finetuned_on == None as those are base models
cross_task_df = cross_task_df[cross_task_df['finetuned_on'].notna()]
cross_task_df = cross_task_df[cross_task_df['task'].isin(focused_tasks)]
cross_task_df = cross_task_df[cross_task_df['finetuned_on'].isin(focused_tasks)]

cross_task_df = cross_task_df.sort_values(by=['finetuned_on', 'task', 'method', 'wbit', 'sparsity', 'group_size'])
cross_task_df.to_csv('results/cross_task_eval_results.csv', index=False)
