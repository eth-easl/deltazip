import os
import json
import pandas as pd
from typing import Callable


def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024


cache_folder = os.environ.get("YAO_CACHE")
OUTPUT_DIR = os.path.join(
    cache_folder, "experiments", "fmzip", "generation_results_reproduce"
)
MODELS_DIR = os.path.join(
    cache_folder, "experiments", "fmzip", "compressed_models_reproduce"
)
# in megabytes
base_model_size = {"open_llama_3b_v2": 6540.97, "pythia-2.8b-deduped": 5297.78}


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
    return gt[0], pred[0][: len(gt[0])]


def judge_em_evaluate(
    data, gt_field: str, pred_field: str, pred_preprocess: Callable = None
):
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
    return {"exact_match": sum(results) / len(results)}


if __name__ == "__main__":
    eval_results = []
    for base_model in [x for x in os.listdir(OUTPUT_DIR)]:
        for task in os.listdir(os.path.join(OUTPUT_DIR, base_model)):
            for config in os.listdir(os.path.join(OUTPUT_DIR, base_model, task)):
                filename = config.removesuffix(".jsonl")
                method = filename.split("_")[-1]
                compression_config = filename.split("_")[0]
                with open(
                    os.path.join(OUTPUT_DIR, base_model, task, config), "r"
                ) as fp:
                    data = [json.loads(x) for x in fp.readlines()][1:]
                    eval_res = judge_em_evaluate(
                        data, "output", "raw_prediction", pred_preprocess=_preprocess
                    )
                    task_name = task.split("-")[0]
                    task_step = task.split("-")[1]
                    eval_results.append(
                        {
                            "base_model": base_model,
                            "task": task_name,
                            "step": task_step,
                            "method": method,
                            "compression_config": compression_config,
                            "eval_res": eval_res["exact_match"],
                            "compression ratio": base_model_size[base_model]
                            / get_size(
                                os.path.join(
                                    MODELS_DIR,
                                    f"{base_model}-{compression_config}_{method}",
                                    task_name,
                                    task_step,
                                )
                            ),
                        }
                    )
    df = pd.DataFrame(eval_results)
    df.to_csv("repro_eval_results.csv", index=False)
