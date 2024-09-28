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

def auto_evaluate(traj_dir: str):
    pass

def get_dir_path_from_id(parent_path: str, id: int):
    subdirs = [x for x in Path(parent_path).iterdir() if x.is_dir()]
    for subdir in subdirs:
        subdir_name = subdir.name
        task_name = subdir_name.split("_")[-3]
        task_id = int(task_name.split(".")[1])
        if task_id == id:
            return f"{parent_path}/{subdir_name}"
    return None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# task id of task with a start url of main page of the website
# these tasks shouldn't require llm eval, would be best if string match eval which is the fastest
def get_start_task_id(website):
    if website == "shopping":
        return 141
    elif website == "shopping_admin":
        return 0
    elif website == "gitlab":
        return 132
    elif website == "reddit":
        return 27
    elif website == "map":
        return 7
    else:
        raise ValueError("Invalid website")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--start_id", type=int, default=0, help="Starting task id")
    parser.add_argument("--end_id", type=int, default=812, help="Ending task id")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of times to run the same task")
    parser.add_argument("--skill_root_path", type=str, default="src/agentlab/skills", help="Root path to save the learned skills")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13", help="Model name to use for inference")
    parser.add_argument("--result_dir", type=str, default="/home/ytliu/agentlab_results/agentlab_baseline", help="Directory to save the results")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps to take for each task.")
    parser.add_argument("--max_exploration_steps", type=int, default=30, help="Maximum number of steps to take for each task.")
    parser.add_argument("--result_dir_id", type=str, default="", help="ID of the result directory")
    parser.add_argument("--learn_dynamics_from_failure", type=str2bool, default=False, help="Whether to learn dynamics from failure")
    parser.add_argument("--eval_metric", type=str, choices=["gt", "auto", "num_steps"], default="gt", help="Evaluation metric to use for intermediate evaluation")
    parser.add_argument("--use_dynamics", type=str2bool, default=True, help="Whether to use dynamics")
    parser.add_argument("--use_screenshot", type=str2bool, default=False, help="Whether to use screenshot")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.result_dir_id == "":
        result_dir_id = f"annotate_"+time.strftime("%Y%m%d%H%M%S", time.localtime())
    else:
        result_dir_id = args.result_dir_id

    config_files = [
        os.path.join("src/agentlab/config_files", f) for f in os.listdir("src/agentlab/config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    # filter the config files based on the website
    config_flags = [config["sites"][0] == args.website for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]

    # init the skill file
    if not os.path.exists(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json"):
        with open(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", "w") as f:
            json.dump([], f)

    start_task_id = get_start_task_id(args.website)
    
    # # run random exploration for the start task
    # for i in range(args.num_samples):
    #     process = Popen([
    #         "python", "src/agentlab/explore.py",
    #         "--task", f"webarena.{start_task_id}",
    #         "--result_dir", f"results/{result_dir_id}/webarena.{start_task_id}",
    #         "--model_name", "azureopenai/"+args.model,
    #         "--id", str(i),
    #         "--max_steps", str(args.max_exploration_steps),
    #         "--use_screenshot", "1"
    #     ])
    #     process.wait()
    #     pass
    
    # distill the skills from the annotations
    for i in range(args.num_samples):
        task_dir = f"results/{result_dir_id}/webarena.{start_task_id}/{i}"
        navi_skills = extract_navi_skill(args.website, task_dir, args.model, args.skill_root_path, result_dir_id, no_goal=True, max_steps=args.max_steps)
        print("*"*50, f"Extracted dynamics from task", "*"*50)
        print(navi_skills)
        save_skills(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", navi_skills)
        general_skills = extract_skills(args.website, task_dir, args.model, args.skill_root_path, result_dir_id, no_goal=True, max_steps=args.max_steps)
        print("*"*50, f"Extracted skills from task", "*"*50)
        print(general_skills)
        save_skills(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", general_skills)

if __name__ == "__main__":
    main()
