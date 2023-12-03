import os
import json
import pandas as pd
from typing import Callable

cache_folder = os.environ.get("YAO_CACHE")
OUTPUT_DIR = os.path.join(
    cache_folder, "experiments", "fmzip", "generation_results_reproduce"
)

def postprocess_pred(text):
    text = text.strip()
    text = text.strip("\\n")
    # logic:
    # if starts with \n, take the remaining
    if text.startswith("\n"):
        text = text.split("\n")[1]
    # if there's \n left, take the first part
    text = text.split("\n")[0]
    return text

def _preprocess(gt, pred):
    pred = [postprocess_pred(p) for p in pred]
    # return the first N characters of the prediction, where N =  the length of the ground truth
    return gt[0], pred[0][:len(gt[0])]

def judge_em_evaluate(data, gt_field: str, pred_field: str, pred_preprocess: Callable=None):
    """
    :param data: a list of data
    :param gt_field: the field name of the ground truth
    :param pred_field: the field name of the prediction
    :param pred_preprocess: a function that preprocesses the prediction
    :return: a dictionary of evaluation results
    """
    def _evaluate_single(datum):
        gt = datum[gt_field]
        pred = datum[pred_field]
        if pred_preprocess is not None:
            gt, pred = pred_preprocess(gt, pred)
        return gt == pred
    
    results = [_evaluate_single(datum) for datum in data]
    return {
        "exact_match": sum(results) / len(results)
    }

if __name__=="__main__":
    eval_results = []
    for base_model in [x for x in os.listdir(OUTPUT_DIR)]:
        for task in os.listdir(os.path.join(OUTPUT_DIR, base_model)):
            for config in os.listdir(os.path.join(OUTPUT_DIR, base_model, task)):
                filename = config.removesuffix(".jsonl")
                method = filename.split("_")[-1]
                compression_config = filename.split("_")[0]
                with open(os.path.join(OUTPUT_DIR, base_model, task, config), "r") as fp:
                    data = [json.loads(x) for x in fp.readlines()][1:]
                    eval_res = judge_em_evaluate(
                        data,
                        "output",
                        "raw_prediction",
                        pred_preprocess = _preprocess
                    )
                    eval_results.append({
                        "base_model": base_model,
                        "task": task,
                        "method": method,
                        "compression_config": compression_config,
                        "eval_res": eval_res['exact_match'],
                    })
    df = pd.DataFrame(eval_results)
    df.to_csv("repro_eval_results.csv", index=False)