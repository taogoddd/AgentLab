import argparse
import json
from agentlab.utils.utils import get_template_task_id_mapping, get_trajectory_from_annotation, save_skills, navi_to_general_skills
from agentlab.extract_navi import extract_navi_skill
from agentlab.extract_skills import extract_skills
from subprocess import Popen
import time
import os


def gt_evaluate(result_dir: str, task_name: str, id: int):
    summary_path = os.path.join(result_dir, f"webarena.{task_name}", str(id), "summary_info.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    reward = summary["cum_reward"]
    if reward == 1.0:
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--start_id", type=int, default=0, help="Starting task id")
    parser.add_argument("--end_id", type=int, default=812, help="Ending task id")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of times to run the same task")
    parser.add_argument("--skill_root_path", type=str, default="src/agentlab/skills", help="Root path to save the learned skills")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name to use for inference")
    return parser.parse_args()

def main():
    args = parse_args()

    id = time.strftime("%Y%m%d%H%M%S", time.localtime())

    if not os.path.exists(f"results/{id}"):
        os.makedirs(f"results/{id}")

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
        if template_id < args.start_id or template_id > args.end_id:
            continue
        
        solved_task_id = None
        for task_id in task_ids:
            num_samples = 1 if solved_task_id is not None else args.num_samples
            for i in range(num_samples):
                # TODO: use multiprocessing to parallelize the task
                # infer on the task with sa_agent
                process = Popen([
                    "python", "src/agentlab/run.py", 
                    "--task", f"webarena.{task_id}",
                    "--result_dir", f"results/{id}/webarena.{task_id}",
                    "--model_name", "openai/gpt-4o",
                    "--id", str(i),
                    "--website", args.website,
                    "--skill_path", f"{args.skill_root_path}/{args.website}/skills.json"
                    "--max_steps", 30
                ])
                process.wait()

                # evaluate the current run of current task w/ gt
                if gt_evaluate(result_dir=f"results/{id}", task_name=task_id, id=i) and solved_task_id is None:
                    solved_task_id = task_id
                    traj_path = f"results/{id}/webarena.{task_id}/{i}"
                    # learn and save navi skill
                    navi_skills = extract_navi_skill(website=args.website, traj_path=traj_path, model=args.model)
                    save_skills(navi_skills, f"{args.skill_root_path}/{args.website}/skills.json")
                    # learn general skills
                    general_skills = extract_skills(website=args.website, traj_path=traj_path, model=args.model, skill_root_path=args.skill_root_path)
                    save_skills(general_skills, f"{args.skill_root_path}/{args.website}/skills.json")
                    break

if __name__ == "__main__":
    main()
    # navi_skills = extract_navi_skill(website="reddit", traj_path="/home/ubuntu/github/AgentLab/results/20240916160632/webarena.29/1", model="gpt-4o")
    # print(navi_skills)