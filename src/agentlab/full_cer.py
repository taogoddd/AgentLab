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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--start_id", type=int, default=0, help="Starting task id")
    parser.add_argument("--end_id", type=int, default=812, help="Ending task id")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of times to run the same task")
    parser.add_argument("--skill_root_path", type=str, default="src/agentlab/skills", help="Root path to save the learned skills")
    parser.add_argument("--model", type=str, default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", help="Model name to use for inference")
    parser.add_argument("--result_dir", type=str, default="/home/ytliu/agentlab_{args.root_result_dir}/agentlab_baseline", help="Directory to save the {args.root_result_dir}")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps to take for each task.")
    parser.add_argument("--result_dir_id", type=str, default=None, help="ID of the result directory")
    parser.add_argument("--learn_dynamics_from_failure", type=str2bool, default=False, help="Whether to learn dynamics from failure")
    parser.add_argument("--eval_metric", type=str, choices=["gt", "auto", "num_steps"], default="gt", help="Evaluation metric to use for intermediate evaluation")
    parser.add_argument("--use_dynamics", type=str2bool, default=True, help="Whether to use dynamics")
    parser.add_argument("--use_screenshot", type=str2bool, default=False, help="Whether to use screenshot")
    parser.add_argument("--root_result_dir", type=str, default="results", help="Root directory to save the results")
    parser.add_argument("--offline_skills_dir", type=str, default="", help="The root directory of the offline skills, each file in the directory is a skill file")
    parser.add_argument("--offline_only", type=str2bool, default=False, help="Whether to run in offline only mode")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.result_dir_id is None:
        if args.offline_skills_dir:
            if args.offline_only:
                result_dir_id = f"offline"+time.strftime("%Y%m%d%H%M%S", time.localtime())
            else:
                result_dir_id = f"offline_online"+time.strftime("%Y%m%d%H%M%S", time.localtime())
        else:
            result_dir_id = f"online"+time.strftime("%Y%m%d%H%M%S", time.localtime())
    else:
        result_dir_id = args.result_dir_id
    # args.skill_root_path /{args.website}/skills_{args.result_dir_id}= args.args.skill_root_path
# /{args.website}/skills_{args.result_dir_id}    # if args.skill_root_path /{args.website}/skills_{args.result_dir_id}is None:
    #     args.skill_root_path /{args.website}/skills_{args.result_dir_id}= f"src/agentlab/skills/{args.website}/skills_{result_dir_id}.json"

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
    
    # if offline skill dir is provided, load the skills and write to the current skill file
    # if args.offline_skills_dir:
    #     for file in os.listdir(args.offline_skills_dir):
    #         with open(f"{args.offline_skills_dir}/{file}", "r") as f:
    #             offline_skills = json.load(f)
    #         with open(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", "r") as f:
    #             skills = json.load(f)
    #         skills.extend(offline_skills)
    #         with open(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", "w") as f:
    #             json.dump(skills, f, indent=4)

    # if offline skills dir is provided, load the last skill file and write to the current skill file
    if args.offline_skills_dir:
        # get the last skill file
        skill_files = [f for f in os.listdir(args.offline_skills_dir) if f.endswith(".json")]
        skill_files = sorted(skill_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        last_skill_file = skill_files[-1]
        with open(f"{args.offline_skills_dir}/{last_skill_file}", "r") as f:
            offline_skills = json.load(f)
        with open(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", "r") as f:
            skills = json.load(f)
        if skills == []:
            skills.extend(offline_skills)
        with open(f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json", "w") as f:
            json.dump(skills, f, indent=4)

    # get task id between start_id and end_id
    new_task_ids = [tid for tid in task_ids if args.start_id <= tid < args.end_id]
    for task_id in new_task_ids:
        try:
            # run only one sample for this setting
            process = Popen([
                "python", "src/agentlab/run.py", 
                "--task", f"webarena.{task_id}",
                "--result_dir", f"{args.root_result_dir}/{result_dir_id}/webarena.{task_id}",
                "--model_name", "openai/"+args.model,
                "--skill_path", f"{args.skill_root_path}/{args.website}/skills_{result_dir_id}.json",
                "--id", "0",
                "--max_steps", str(args.max_steps),
                "--use_screenshot", "1" if args.use_screenshot else "0"
            ])
            process.wait()
            pass
            
            if not args.offline_only:
                # 0 here for later possible samplings
                task_dir = f"{args.root_result_dir}/{result_dir_id}/webarena.{task_id}/0"

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
