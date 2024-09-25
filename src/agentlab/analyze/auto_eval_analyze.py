import json
from pathlib import Path
import pandas as pd
from typing import Union
from subprocess import Popen
from tqdm import tqdm
import argparse

def get_auto_eval_accuracy(results_path: str, model_name: str, prompt_type: str):
    '''
    results_path: str
        The path to the directory containing the results of the experiments. (result root dir)
    model_name: str
        can only be one of the following: ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
    prompt_type: str
        can only be one of the following: ["text", "vision"]
    '''
    subdirs = [x for x in Path(results_path).iterdir() if x.is_dir()]
    records = {}
    for subdir in tqdm(subdirs):
        subdir_name = subdir.name
        id = int(subdir_name.split(".")[1])
        print(f"Processing id: {id}")
        # if (subdir / "summary_info.json").exists():
        #     with open(subdir / "summary_info.json", "r") as f:
        #         summary_info = json.load(f)
        #     score = summary_info["cum_reward"]
        #     records[id] = score
        
        # run auto evaluation
        # try:
        process = Popen([
            "python", "-m", "agentlab.autoeval.evaluate_trajectory",
            "--result_dir", f"{results_path}/{subdir_name}/0",
            "--model", model_name,
            "--prompt", prompt_type
        ])
        process.wait()

        # read the autoeval result
        with open(f"{results_path}/{subdir_name}/0/{model_name}_autoeval.json", "r") as f:
            autoeval_result = json.load(f)
            autoeval_result = autoeval_result[0]
        gt = autoeval_result["gt"]
        auto_eval = autoeval_result["rm"]
        print(f"GT: {gt}, Auto Eval: {auto_eval}")
        records[id] = (gt, auto_eval)
        # except Exception as e:
        #     gt = None
        #     auto_eval = "Error"
        #     records[id] = (gt, auto_eval)
        #     print(f"Error: {e}")
        
        # print current average accuracy
        accuracy = sum([1 for id, (gt, auto_eval) in records.items() if auto_eval != "Error" and gt == auto_eval]) / len(records)
        print(f"Current accuracy: {accuracy}")
    
    print(records)
    # print error ids if any
    error_ids = [id for id, (gt, auto_eval) in records.items() if auto_eval == "Error"]
    print(f"Error ids: {error_ids}")
    # get the accuracy
    accuracy = sum([1 for id, (gt, auto_eval) in records.items() if auto_eval != "Error" and gt == auto_eval]) / len(records)
    print(f"Total Accuracy: {accuracy}")
    return accuracy, records

# get_auto_eval_accuracy("/home/ubuntu/github/AgentLab/results/streaming_single_action_merged_skills_all_dynamics_temp_0.1_no_hints_not_ldff20240922111436", "gpt-4o", "vision")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the result trajectory directory, e.g., 'xxx/webarena.0'.")
    # autoeval
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o"])
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "vision"])

    args = parser.parse_args()

    get_auto_eval_accuracy(args.results_dir, args.model, args.prompt)