import argparse
import json
from agentlab.utils.utils import get_template_task_id_mapping, get_trajectory_from_annotation, save_skills, navi_to_general_skills
from agentlab.extract_navi import extract_navi_skill
from agentlab.extract_skills import extract_skills
import torch
from subprocess import Popen
import time
import os
from pathlib import Path
import subprocess
from visualwebarena.evaluation_harness import image_utils
from agentlab.new_run_v import run as run_v_run

eval_captioning_model = "Salesforce/blip2-flan-t5-xl"
eval_captioning_model_device = "cuda" if torch.cuda.is_available() else "cpu"
captioning_fn = image_utils.get_captioning_fn(
    "cuda" if torch.cuda.is_available() else "cpu",
    torch.float16
    if (
        torch.cuda.is_available()
        and eval_captioning_model_device == "cuda"
    )
    else torch.float32,
    eval_captioning_model,
)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "reddit", "classifieds"])
    parser.add_argument("--start_id", type=int, default=None, help="Starting task id")
    parser.add_argument("--end_id", type=int, default=None, help="Ending task id, not greater than max_id in the website tasks")
    parser.add_argument("--skill_root_path", type=str, default="src/agentlab/skills", help="Root path to save the learned skills")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13", help="Model name to use for inference")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps to take for each task.")
    parser.add_argument("--result_dir_id", type=str, default="", help="ID of the result directory")
    parser.add_argument("--learn_dynamics_from_failure", type=str2bool, default=False, help="Whether to learn dynamics from failure")
    parser.add_argument("--eval_metric", type=str, choices=["gt", "auto", "num_steps"], default="gt", help="Evaluation metric to use for intermediate evaluation")
    parser.add_argument("--use_dynamics", type=str2bool, default=True, help="Whether to use dynamics")
    parser.add_argument("--use_screenshot", type=str2bool, default=True, help="Whether to use screenshot")
    parser.add_argument("--use_ax_tree", type=str2bool, default=False, help="Whether to use ax tree")
    return parser.parse_args()

def main():
    args = parse_args()
    args = parse_args()

    if args.result_dir_id == "":
        result_dir_id = f"v_streaming_single_action_merged_skills_all_dynamics_temp_0.1_no_hints{'_not_ldff' if not args.learn_dynamics_from_failure else ''}"+time.strftime("%Y%m%d%H%M%S", time.localtime())
    else:
        result_dir_id = args.result_dir_id

    config_list = json.load(open("visualwebarena/test_raw.json", "r"))
    # filter the config files based on the website
    config_website_flags = [config["sites"] == [args.website] for config in config_list]
    config_reset_flags = [config["require_reset"] for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_website_flags) if flag]

    # init the skill file
    if not os.path.exists(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json"):
        with open(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", "w") as f:
            json.dump([], f)

    # get task id between start_id and end_id
    if args.start_id is not None and args.end_id is not None:
        new_task_ids = [tid for tid in task_ids if args.start_id <= tid < args.end_id]
    elif args.start_id is not None:
        new_task_ids = [tid for tid in task_ids if tid >= args.start_id]
    elif args.end_id is not None:
        new_task_ids = [tid for tid in task_ids if tid < args.end_id]
    else:
        new_task_ids = task_ids

    for task_id in new_task_ids:
        try:
            config = [config for config in config_list if config["task_id"] == task_id][0]
            # reset env if required
            reset_flag = config["require_reset"]

            page_eval_flag = True if "page_image_query" in config["eval"].get("eval_types", []) else False

            if reset_flag:
                # run bash command to reset the environment
                if args.website == "shopping":
                    subprocess.run(["src/agentlab/v_reset_scripts/reset_shopping.sh"])
                elif args.website == "reddit":
                    subprocess.run(["src/agentlab/v_reset_scripts/reset_reddit.sh"])

            # Prepare arguments for run_v.py
            args_run_v = argparse.Namespace()
            args_run_v.model_name = "azureopenai/" + args.model
            args_run_v.task_name = f"visualwebarena.{task_id}"
            args_run_v.start_url = "https://www.google.com"  # Adjust if needed
            args_run_v.slow_mo = 30  # Default or adjust as needed
            args_run_v.headless = True  # Set based on your requirements
            args_run_v.demo_mode = True  # Set based on your requirements
            args_run_v.use_html = False  # Set based on your requirements
            args_run_v.use_ax_tree = args.use_ax_tree
            args_run_v.use_screenshot = args.use_screenshot
            args_run_v.multi_actions = True  # Set based on your requirements
            args_run_v.action_space = "bid"  # Set based on your requirements
            args_run_v.use_history = True  # Set based on your requirements
            args_run_v.use_thinking = True  # Set based on your requirements
            args_run_v.use_reminder = False  # Set based on your requirements
            args_run_v.max_steps = args.max_steps
            args_run_v.skill_path = f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json"
            args_run_v.result_dir = f"results/{result_dir_id}/webarena.{task_id}"
            args_run_v.id = "0"
            args_run_v.website = args.website
            args_run_v.max_skills = 10  # Set based on your requirements

            if page_eval_flag:
                args_run_v.max_steps = 20

            # Call the run function directly
            run_v_run(args_run_v, captioning_fn=captioning_fn)

            # 0 here for later possible samplings
            task_dir = f"results/{result_dir_id}/webarena.{task_id}/0"

            # run autoeval if args.eval_metric is auto
            if args.eval_metric == "auto":
                process = Popen([
                    "python", "-m", "agentlab.autoeval.evaluate_trajectory",
                    "--result_dir", task_dir,
                    "--model", "gpt-4o",
                ])
                process.wait()
            
            elif args.eval_metric == "num_steps":
                with open(f"{task_dir}/summary_info.json", "r") as f:
                    summary = json.load(f)
                num_steps = summary["stats.cum_steps"]
                if num_steps < 20:
                    eval = True
                else:
                    eval = False
            elif args.eval_metric == "gt":
                eval = gt_evaluate(f"{task_dir}/summary_info.json")
            else:
                eval = auto_evaluate(task_dir)

            if args.use_dynamics and (args.learn_dynamics_from_failure or eval):
                # extract dynamics from all the tasks
                navi_skills = extract_navi_skill(args.website, task_dir, args.model, args.skill_root_path, result_dir_id)
                print("*"*50, f"Extracted dynamics from task", "*"*50)
                print(navi_skills)
                save_skills(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", navi_skills)

            # extract general skills from the tasks that are solved
            if eval:
                general_skills = extract_skills(args.website, task_dir, args.model, args.skill_root_path, result_dir_id)
                print("*"*50, f"Extracted skills", "*"*50)
                print(general_skills)
                save_skills(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", general_skills)

        except Exception as e:
            print(e)
            continue
if __name__ == "__main__":
    main()
