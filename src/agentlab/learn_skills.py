import argparse
import json
from agentlab.utils.utils import get_template_task_id_mapping, get_trajectory_from_annotation, save_skills, navi_to_general_skills
from agentlab.extract_navi import extract_navi_skill
from agentlab.extract_skills import extract_skills
from subprocess import Popen
import time
import os
from pathlib import Path

def gt_evaluate(summary_path: str):
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    reward = summary["cum_reward"]
    if reward == 1.0:
        return True
    return False

def get_dir_path_from_id(parent_path: str, id: int):
    subdirs = [x for x in Path(parent_path).iterdir() if x.is_dir()]
    for subdir in subdirs:
        subdir_name = subdir.name
        task_name = subdir_name.split("_")[-3]
        task_id = int(task_name.split(".")[1])
        if task_id == id:
            return f"{parent_path}/{subdir_name}"
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--start_id", type=int, default=0, help="Starting task id")
    parser.add_argument("--end_id", type=int, default=812, help="Ending task id")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of times to run the same task")
    parser.add_argument("--skill_root_path", type=str, default="src/agentlab/skills", help="Root path to save the learned skills")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name to use for inference")
    parser.add_argument("--result_dir", type=str, default="/home/ubuntu/agentlab_results/agentlab_baseline", help="Directory to save the results")
    return parser.parse_args()

def main():
    args = parse_args()

    id = "baseline_reattempts_single_action"+time.strftime("%Y%m%d%H%M%S", time.localtime())

    config_files = [
        os.path.join("src/agentlab/config_files", f) for f in os.listdir("src/agentlab/config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    # filter the config files based on the website
    filtered_config_files = [config for config in config_list if config["sites"][0] == args.website]
    template_task_id_mapping = get_template_task_id_mapping(filtered_config_files)

    for template_id, task_ids in template_task_id_mapping.items():

        # check if any of the tasks in the template have been solved
        solved_task_id = None
        solved_task_path = None
        for task_id in task_ids:
            path = get_dir_path_from_id(args.result_dir, task_id)
            if gt_evaluate(f"{path}/summary_info.json"):
                solved_task_id = task_id
                solved_task_path = path
                break
        
        if solved_task_id is not None and solved_task_id >= args.start_id and solved_task_id < args.end_id:
            # get the intent from the solved task
            intent = get_trajectory_from_annotation(f"{solved_task_path}")[0]["obs"]["goal"]
            # extract the skills from the solved task
            navi_skills = extract_navi_skill(args.website, solved_task_path, args.model)
            print("*"*50, f"Extracted dynamics from task: '{intent}'", "*"*50)
            print(navi_skills)
            save_skills(f"{args.skill_root_path}/{args.website}/skills.json", navi_skills)
            general_skills = extract_skills(args.website, solved_task_path, args.model, args.skill_root_path)
            print("*"*50, f"Extracted skills from {intent}", "*"*50)
            print(general_skills)
            save_skills(f"{args.skill_root_path}/{args.website}/skills.json", general_skills)
            
            # attempt to solve the other tasks in the template
            for task_id in task_ids:
                if task_id == solved_task_id or task_id < args.start_id or task_id > args.end_id or gt_evaluate(f"{get_dir_path_from_id(args.result_dir, task_id)}/summary_info.json"):
                    continue
                num_samples = args.num_samples
                for i in range(num_samples):
                    # infer on the task with sa_agent
                    process = Popen([
                        "python", "src/agentlab/run.py", 
                        "--task", f"webarena.{task_id}",
                        "--result_dir", f"results/{id}/webarena.{task_id}",
                        "--model_name", "openai/gpt-4o",
                        "--skill_path", f"{args.skill_root_path}/{args.website}/skills.json",
                        "--id", str(i),
                    ])
                    process.wait()

if __name__ == "__main__":
    main()
