import os
import argparse
import json
from browsergym.experiments.loop import EnvArgs, ExpArgs
from agentlab.agents.skill_augmented_agent.sa_agent import SkillAugmentedAgentArgs
from agentlab.agents.skill_augmented_agent.sa_agent_prompt import SkillAugmentedPromptFlags
from agentlab.llm.chat_api import ChatModelArgs, OpenAIChatModelArgs, AzureOpenAIChatModelArgs
from agentlab.agents.dynamic_prompting import Flags
import agentlab.agents.dynamic_prompting as dp
from agentlab.select_skills import select_skills
from pathlib import Path
import torch
from visualwebarena.evaluation_harness import image_utils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-4o",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--slow_mo", type=int, default=30, help="Slow motion delay for the playwright actions."
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=True,
        help="Run the experiment in headless mode (hides the browser windows).",
    )
    parser.add_argument(
        "--demo_mode",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html", type=str2bool, default=False, help="Use HTML in the agent's observation space."
    )
    parser.add_argument(
        "--use_ax_tree",
        type=str2bool,
        default=True,
        help="Use AX tree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )
    parser.add_argument(
        "--multi_actions", type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="bid",
        choices=["python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"],
        help="",
    )
    parser.add_argument(
        "--use_history",
        type=str2bool,
        default=True,
        help="Use history in the agent's observation space.",
    )
    parser.add_argument(
        "--use_thinking",
        type=str2bool,
        default=True,
        help="Use thinking in the agent (chain-of-thought prompting).",
    )
    parser.add_argument(
        "--use_reminder",
        type=str2bool,
        default=False,
        help="Use reminder in the agent's observation space.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Maximum number of steps to take for each task.",
    )
    parser.add_argument(
        "--skill_path",
        type=str,
        default=None,
        help="Path to the skill file to load for the agent.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Path to the directory to save the experiment results.",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="ID of the task to run, indicating the times of the task has been run.",
    )
    parser.add_argument(
        "--website",
        type=str,
        default="shopping",
        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"],
        help="Website to run the task"        
    )
    parser.add_argument(
        "--max_skills",
        type=int,
        default=10,
        help="Maximum number of skills to select for the agent."
    )

    return parser.parse_args()

def format_skills(skills):
    navi_skills = [skill for skill in skills if skill["type"] == "navi"]
    general_skills = [skill for skill in skills if skill["type"] == "general"]
    navi_skills_str = "# Shortcut URLs to navigate to a specific page, you can use goto(URL: str) to directly navigate to these pages if needed:\n"
    for i, skill in enumerate(navi_skills):
        navi_skills_str += f"## Page {i+1}: {skill['name']}\n"
        navi_skills_str += f"### Page description: {skill['description']}\n"
        navi_skills_str += f"### Page possible usages: {skill['usages']}\n"
        navi_skills_str += f"### Page URL: {skill['URL']}\n"

    general_skills_str = "# General skills:\n"
    for i, skill in enumerate(general_skills):
        general_skills_str += f"## Skill {i+1}: {skill['skill']}\n"
        general_skills_str += f"{skill['steps']}\n"
    
    if len(navi_skills) == 0:
        navi_skills_str = "No shortcuts available."

    if len(general_skills) == 0:
        general_skills_str = "No general skills available."
    
    skills_str = f"{navi_skills_str}\n{general_skills_str}"
    
    return skills_str

def merged_format_skills(skills):
    prefix = "# Skills (common workflows summarized from previous experience):\n"

    for i, skill in enumerate(skills):
        if skill["type"] == "navi":
            prefix += f"## Skill {i+1}: navigate to {skill['name']}\n"
            prefix += f"### {skill['name']} contents: {skill['description']}\n"
            prefix += f"### Potential usages: {skill['usages']}\n"
            prefix += f"### Steps:\n1. navigate to {skill['name']}.```goto('{skill['URL']}')```\n"
        elif skill["type"] == "general":
            prefix += f"## Skill {i+1}: {skill['skill']}\n"
            prefix += f"### Steps:\n{skill['steps']}\n"
    
    if len(skills) == 0:
        prefix = "No skills available."
    
    return prefix

def run(args, captioning_fn=None):
    task_id = args.task_name.split(".")[1]
    configs_path = os.path.join("visualwebarena", f"test_raw.json")
    configs = json.load(open(configs_path))
    config = [c for c in configs if c["task_id"] == int(task_id)][0]
    intent = config["intent"]
    start_url = config["start_url"]

    goal_image_urls = config.get("image", [])
    if goal_image_urls is None:
        goal_image_urls = []
    if isinstance(goal_image_urls, str):
        goal_image_urls = [goal_image_urls]
    
    def remove_placeholders_in_url(url):
        mapping ={
            "__CLASSIFIEDS__": "CLASSIFIEDS",
            "__REDDIT__": "REDDIT",
            "__SHOPPING__": "SHOPPING",
            "__WIKIPEDIA__": "WIKIPEDIA",
        }
        for key, value in mapping.items():
            if key in url:
                base_url = os.environ.get(value)
                url = url.replace(key, base_url)
        return url

    goal_image_urls = [remove_placeholders_in_url(url) for url in goal_image_urls]

    # get the skills
    # skills = json.load(open(args.skill_path))
    # if len(skills) > args.max_skills:
    #     retrieved_skills = select_skills(intent, goal_image_urls, args.website, args.skill_path, args.model_name.split("/")[-1], (args.max_skills//2, args.max_skills-args.max_skills//2))
    # else:
    #     retrieved_skills = skills
    # print("*"*50, "Retrieved skills", "*"*50)
    # print(retrieved_skills)
    # skill_str = merged_format_skills(retrieved_skills)
    
    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        # viewport={"width": 1280, "height": 2048},
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    if args.model_name.startswith("openai"):
        chat_model_args = OpenAIChatModelArgs(
            model_name=args.model_name,
            max_total_tokens=128_000,
            max_input_tokens=126_000,
            max_new_tokens=2_000,
            temperature=0.1,
            vision_support=True,
        )
    elif args.model_name.startswith("azure"):
        chat_model_args = AzureOpenAIChatModelArgs(
            model_name=args.model_name,
            max_total_tokens=128_000,
            max_input_tokens=126_000,
            max_new_tokens=2_000,
            temperature=0.1,
            vision_support=True,
        )
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=SkillAugmentedAgentArgs(
            chat_model_args=chat_model_args,
            flags = SkillAugmentedPromptFlags(
                obs=dp.ObsFlags(
                    use_html=False,
                    use_ax_tree=args.use_ax_tree,
                    use_focused_element=True,
                    use_error_logs=True,
                    use_history=False,
                    use_past_error_logs=False,
                    use_action_history=False,
                    use_think_history=False,
                    use_diff=False,
                    html_type="pruned_html",
                    use_screenshot=args.use_screenshot,
                    use_som=True,
                    extract_visible_tag=True,
                    extract_clickable_tag=True,
                    extract_coords="False",
                    filter_visible_elements_only=False,
                    # openai_vision_detail="high"
                ),
                action=dp.ActionFlags(
                    multi_actions=False,
                    # change action space here
                    action_set="wa_base",
                    long_description=True,
                    individual_examples=True,
                ),
                use_plan=False,
                use_criticise=False,
                use_thinking=True,
                use_reminder=False,
                use_memory=False,
                use_concrete_example=True,
                use_abstract_example=True,
                use_hints=True,
                enable_chat=False,
                max_prompt_tokens=None,
                be_cautious=False,
                extra_instructions=None,
                skill_str="",
            ),
        ),
        captioning_fn=captioning_fn,
    )

    exp_args.prepare(Path(args.result_dir))
    exp_args.run()

    os.rename(exp_args.exp_dir, f"{args.result_dir}/{args.id}")

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()