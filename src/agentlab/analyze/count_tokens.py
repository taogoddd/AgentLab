import json
import tiktoken
from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_trajectory, get_trajectory_from_annotation, get_website_name_from_url
import os
from pathlib import Path
from tqdm import tqdm

def count_multimodal_messages_tokens(messages, model="gpt-4o") -> int:
    token_count = 0
    for message in messages:
        if "content" in message:
            message = message["content"]

        if isinstance(message, str):
            token_count += count_tokens(message, model)
        # handles messages with image content
        elif isinstance(message, (list, tuple)):
            for part in message:
                if not isinstance(part, dict):
                    raise ValueError(
                        f"The message is expected to be a list of dicts, but got list of {type(message)}"
                    )
                if part["type"] == "text":
                    token_count += count_tokens(part["text"], model)
                elif part["type"] == "image_url":
                    if part["image_url"].get("detail", None) == "high":
                        token_count += 1500
                    else:
                        token_count += 85
        else:
            raise ValueError(
                f"The message is expected to be a string or a list of dicts, but got {type(message)}"
            )
    return token_count

def count_tokens(text, model="gpt-4o"):
    """Count the number of tokens in a text."""
    if text == None:
        text = ""

    return len(tiktoken.encoding_for_model(model).encode(text))

def count_retrieve_tokens(skill_file_path: str):
    with open(skill_file_path, "r") as f:
        skill_data = json.load(f)
    navi_skills = [skill for skill in skill_data if skill["type"] == "navi"]
    general_skills = [skill for skill in skill_data if skill["type"] == "general"]

    navi_token_count = 0
    general_token_count = 0

    # count tokens for skill1, skill1+skill2, skill1+skill2+skill3, ...
    for i, skill in enumerate(navi_skills):
        # count tokens for this skill and skills before this skill
        for j in range(i+1):
            navi_token_count += count_tokens(navi_skills[j]["page-summary"])
            navi_token_count += count_tokens(navi_skills[j]["description"])
            navi_token_count += count_tokens(navi_skills[j]["usages"])
    
    for i, skill in enumerate(general_skills):
        # count tokens for this skill and skills before this skill
        for j in range(i+1):
            general_token_count += count_tokens(general_skills[j]["skill"])
            general_token_count += count_tokens(general_skills[j]["steps"])
    
    return navi_token_count, general_token_count

def construct_distill_messages(trajectory):
    existing_skills_str = ""
    # goal = trajectory[0]["obs"]["goal"]
    goal_str = ""
    system_prompt = f"""\
You will be given the state-action trajectory of a user interacting with a webpage and the overall goal of the trajectory.
You need to summarize skills from the trajectory.
Skills are a subset of actions that the user takes to achieve a sub-goal.
You should break the overall goal into sub-goals and summarize each sub-goal as a skill.
Represent the non-fixed elements (input text, button strings) and non-fixed words (e.g. a specific forum name / user name; an option) with descriptive variable names as shown in the example.
Also for these elements, you should consider the abstract version of it to make the skill more general. e.g. "filter posts by hottest" -> "filter posts by {{sort criterion}}".

Output format:
<think>
think step by step
</think>
<skill>
skill1 name here.
</skill>
<steps>
The steps of the skill1 here.
</steps>
<think>
think step by step
</think>
<skill>
skill2 name here.
</skill>
<steps>
The steps of the skill2 here.
</steps>
...

# Examples
## Example 1
Overall goal: I want to get the cheapest product in the Cabinets, Racks & Shelves category
Current website:
Existing skills:
Skill 1: Sort products by {{sort criterion}}
1. To sort the products by {{sort criterion}}, I need to click on the "Sort by" dropdown menu.
```click({{sort by id}})```
2. To sort the products by {{sort criterion}}, I need to select the {{sort criterion}} option from the "Sort by" dropdown menu.
```click({{sort criterion id}})```

Human user trajectory (states neglected here): 
Step 0:
<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Home & Kitchen" section, which likely contains the Cabinets, Racks & Shelves category. I will start by hovering over the "Home & Kitchen" menu item to reveal the subcategories.
</think>
<action>
hover('735')
</action>
Step 1:
<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Storage & Organization" section under the "Home & Kitchen" menu. I will start by hovering over the "Storage & Organization" menu item to reveal the subcategories.
</think>
<action>
hover('786')
</action>
Step 2:
<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to click on the "Racks, Shelves & Drawers" link under the "Storage & Organization" menu. This will take me to the desired category.
</think>
<action>
click('797')
</action>
Step 3:
<think>
To find the cheapest product in the Cabinets, Racks & Shelves category, I need to sort the products by price in ascending order. I will start by clicking on the "Sort by" dropdown menu.
</think>
<action>
click('600')
</action>
Step 4:
<think>
To find the cheapest product in the Cabinets, Racks & Shelves category, I need to select the "Price" option from the "Sort by" dropdown menu. This will display the products in order of price.
</think>
<action>
click('601')
</action>
Step 5:
<think>
To find the cheapest product in the Cabinets, Racks & Shelves category, I need to sort the products by price in ascending order but now it is in descending order. I will start by clicking on the arrow icon next to the "Price" option.
</think>
<action>
click('602')
</action>
Step 6:
<think>
The cheapest product in the Cabinets, Racks & Shelves category is "Garage storage shelves" as shown in the product list. I will tell the user about this product.
</think>
<action>
send_msg_to_user('Garage storage shelves')
</action>

Output:
<think>
The user shows how to browse products in the Cabinets, Racks & Shelves category. The abstract version is Browse products in {{sub-category}}. This has not been summarized before. So I will summarize it as a skill.
</think>
<skill>
Browse products in {{sub-category}}
</skill>
<steps>
1. To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Home & Kitchen" section, which likely contains the Cabinets, Racks & Shelves category. I will start by hovering over the "Home & Kitchen" menu item to reveal the subcategories.
```hover({{main category id}})```
2. To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Storage & Organization" section under the "Home & Kitchen" menu. I will start by hovering over the "Storage & Organization" menu item to reveal the subcategories.
```hover({{sub-category id}})```
3. To browse the products in the Cabinets, Racks & Shelves category, I need to click on the "Racks, Shelves & Drawers" link under the "Storage & Organization" menu. This will take me to the desired category.
```click{{sub-sub-category id}}```
</steps>
<think>
The user shows how to sort products by price in ascending order. The abstract version is Sort products by {{sort criterion}}. This, however, has been summarized before. So I will not summarize it again.
</think>
<skill>
Sort products by {{sort criterion}}
</skill>
<steps>
Summarized before
</steps>

IMPORTANT NOTES you should absolutely follow: 
1. DO NOT include any other words except skills and steps as the format stated above.
2. Check existing skills before generating, do not summarize skills that have already been summarized, instead, use "Summarized before" in the steps.
3. You should break the overall goal into sub-goals and summarize each sub-goal as a skill.
"""
    prefix = f"""\
{goal_str}
Current website: 
Exisiting skills: 
{existing_skills_str}
Human user trajectory:
"""
    human_prompt = [
        {
            "type": "text",
            "text": prefix
        }
    ]
    for i, step in enumerate(trajectory):
        obs = step["obs"]
        url = obs["url"]
        processed_obs = step["processed_obs"]
        action = step["action"]
        reward = step["reward"]
        # screenshot_base64 = img_array_to_base64(processed_obs["screenshot"])
        # som_screenshot_base64 = img_array_to_base64(processed_obs["screenshot_som"])
        axtree_str = processed_obs["axtree_txt"]
        human_prompt.append({
            "type": "text",
            "text": f"Step {i}:\nObservation: "
        })
        human_prompt.append({
            "type": "image_url",
            "image_url": {
                "url": f""
            }
        })
        human_prompt.append({
            "type": "text",
            "text": f"URL: {url}"
        })
        human_prompt.append({
            "type": "text",
            "text": f"Action: {action}"
        })
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": human_prompt
        }
    ]
    return messages

def count_distill_tokens(traj_dir_path: str):
    # count the number of dirs in root_traj_dir_path
    subdirs = [x for x in Path(traj_dir_path).iterdir() if x.is_dir()]
    total_tokens = 0
    num_tasks = len(os.listdir(traj_dir_path))
    for subdir in tqdm(subdirs):
        # get the string representation of the task id
        task_dir = f"{str(subdir)}/0"
        trajectory = get_trajectory_from_annotation(task_dir)
        messages = construct_distill_messages(trajectory)
        tokens = count_multimodal_messages_tokens(messages)
        total_tokens += tokens
    return total_tokens

def count_inference_tokens(path: str, model: str = "gpt-4o"):
    subdirs = [x for x in Path(path).iterdir() if x.is_dir()]

    total_tokens = 0
    # parse out the id from the subdirectory name, e.g. get 0 from 2024-06-27_11-41-13_GenericAgent_on_webarena.0_51_14d4f1
    for subdir in tqdm(subdirs):
        subdir_name = subdir.name
        id = int(subdir_name.split(".")[1])
        # check whether file exists
        if (subdir / "0" / "summary_info.json").exists():
            with open(subdir / "0" / "summary_info.json", "r") as f:
                summary_info = json.load(f)
            tokens = summary_info.get("stats.cum_openai_prompt_tokens", 0) # prompt + completion
            total_tokens += tokens
    # if model == "gpt-4o":
    #     # $5 per 1M tokens
    #     cost = total_tokens / 1e6 * 5
    # else:
    #     raise ValueError("Model not supported")
    print(f"Total tokens: {total_tokens}")
    # average_tokens = total_tokens / len(subdirs)
    # print(f"Average tokens per task: {average_tokens}")
    # print(f"Total cost: ${cost}")
    # print(f"Average cost per task: ${cost / len(subdirs)}")
    return total_tokens
    

# we count input tokens only

total_tokens = 0

navi_token_count, general_token_count = count_retrieve_tokens("/home/ytliu/github/AgentLab/src/agentlab/skills/gitlab/skills_streaming_single_action_merged_skills_all_dynamics_temp_0.1_no_hints_not_ldff20240924075014.json")
total_retrieve_tokens = navi_token_count + general_token_count
print(f"Total retrieve tokens: {total_retrieve_tokens}")

total_distill_tokens = count_distill_tokens("/home2/ytliu/webarena/results/cer_results/cer_online_gitlab")
print(f"Total distill tokens: {total_distill_tokens}")

total_inference_tokens = count_inference_tokens("/home2/ytliu/webarena/results/cer_results/cer_online_gitlab")
print(f"Total inference tokens: {total_inference_tokens}")

total_tokens = total_retrieve_tokens + total_distill_tokens + total_inference_tokens
print(f"Total tokens: {total_tokens}")
num_tasks = 196
average_tokens = total_tokens / num_tasks
print(f"Average tokens per task: {average_tokens}")