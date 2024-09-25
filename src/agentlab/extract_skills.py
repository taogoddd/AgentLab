from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_trajectory, get_trajectory_from_annotation, get_website_name_from_url
import os
from webarena.llms.providers.openai_utils import generate_from_openai_chat_completion_with_key_pool
from agentlab.utils.utils import reset_skills
import json
from PIL import Image
import requests
import json
import numpy as np
from agentlab.llm.llm_utils import (
    ParseError,
    count_tokens,
    image_to_jpg_base64_url,
    parse_html_tags_raise,
    extract_code_blocks,
)

# def extract_skill(traj_path: str, skill_root_path: str, website: str):
#     trajectory = get_trajectory_from_annotation(traj_path)

#     skill_type = "sliced_skills"
#     if not os.path.exists(f"{skill_root_path}/{website}/{skill_type}"):
#         os.makedirs(f"{skill_root_path}/{website}/{skill_type}")
#     slicer = Slicer()
#     sliced_skills = slicer.slice(trajectory)
#     slicer.save_as_readable_json(sliced_skills, f"{skill_root_path}/{website}/{skill_type}/skills.json")
#     slicer.save(sliced_skills, f"{skill_root_path}/{website}/{skill_type}/skills.pkl")

#     skill_type = "instructed_skills"
#     if not os.path.exists(f"{skill_root_path}/{website}/{skill_type}"):
#         os.makedirs(f"{skill_root_path}/{website}/{skill_type}")
#     instructor = Instructor()
#     instructed_skills = instructor.instruct(sliced_skills)
#     instructor.save_as_readable_json(instructed_skills, f"{skill_root_path}/{website}/{skill_type}/skills.json")
#     instructor.save(instructed_skills, f"{skill_root_path}/{website}/{skill_type}/skills.pkl")

# extract_skill(
#     traj_path="/home/ubuntu/agentlab_results/2024-09-11_02-01-51_baseline/2024-09-11_02-01-52_GenericAgent_on_webarena.13_2_ab01c0",
#     skill_root_path="/home/ubuntu/github/AgentLab/src/agentlab/skills",
#     website="shopping_admin"
# )


from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_trajectory, get_trajectory_from_annotation, get_website_name_from_url
from agentlab.utils.utils import parse_html_tag_output, get_website_description

# TODO: add website description str

# this desc str is for extracting skills from a trajectory only
def get_skills_desc(skills_path: str):
    # load skills
    with open(skills_path, "r") as f:
        skills = json.load(f)
    if not skills:
        return ["No skills learned yet."]
    skills_str = ""
    for i, skill in enumerate(skills):
        if skill["type"] == "navi":
            skills_str += f"Skill {i+1}: navigate to {skill['name']}\n"
            skills_str += f"Description: {skill['description']}\n"
            skills_str += f"Usages: {skill['usages']}\n"
            skills_str += f"1. ```goto('{skill['URL']}')```\n"
        else:
            skills_str += f"Skill {i+1}: {skill['skill']}\n"
            skills_str += f"{skill['steps']}\n"
    
    return skills_str

    # # format navi skills string
    # navi_skills_str = f""
    # for i, skill in enumerate(navi_skills):
    #     navi_skills_str += f"{i+1}. {skill['name']}\n"
    #     navi_skills_str += f"Description: {skill['description']}\n"
    #     navi_skills_str += f"Usages: {skill['usages']}\n"
    #     navi_skills_str += f"{skill['steps']}\n"
    
    # # format general skills string
    # general_skills_str = f"General skills:\n"

    # for i, skill in enumerate(general_skills):
    #     general_skills_str += f"Skill {i+1}: {skill['skill']}\n"
    #     navi_skills_str += f"{skill['steps']}\n"

def construct_prompt_messages(
        website: str,
        skills_file_path: str,
        trajectory: list[TrajectoryStep],
    ):
    existing_skills_str = get_skills_desc(skills_path=skills_file_path)
    goal = trajectory[0]["obs"]["goal"]
    goal_image_urls = trajectory[0]["obs"].get("goal_image_urls", [])
    img_urls = []
    if goal_image_urls:
        for url in goal_image_urls:
            if url.startswith("http"):
                input_image = Image.open(requests.get(url, stream=True).raw)
            else:
                input_image = Image.open(url)
            img_urls.append(image_to_jpg_base64_url(input_image))
    system_prompt = f"""\
You will be given the state-action trajectory of a user interacting with a webpage and the overall goal of the trajectory.
You need to summarize skills from the trajectory.
Skills are a subset of actions that the user takes to achieve a sub-goal.
You should break the overall goal into sub-goals and summarize each sub-goal as a skill.
Represent the non-fixed elements (input text, button strings) and non-fixed words (e.g. a specific forum name / user name; an option) with descriptive variable names as shown in the example.

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
Current website: {get_website_description("shopping")}
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
    goal = f"""\
Overall goal of the trajectory: {goal}
Images relevant to the goal:
"""
    other_info = f"""\
Current website: {get_website_description(website)}
Existing summarized pages:
{existing_skills_str}
Human user trajectory:
"""
    human_prompt = [
        {
            "type": "text",
            "text": goal
        },
    ]
    for i, img_url in enumerate(img_urls):
        human_prompt.append({
            "type": "image_url",
            "image_url": {
                "url": f"{img_url}"
            }
        })
    human_prompt.append({
        "type": "text",
        "text": other_info
    })
    for i, step in enumerate(trajectory):
        obs = step["obs"]
        url = obs["url"]
        processed_obs = step["processed_obs"]
        action = step["action"]
        reward = step["reward"]
        screenshot_base64 = img_array_to_base64(processed_obs["screenshot"])
        som_screenshot_base64 = img_array_to_base64(processed_obs["screenshot_som"])
        axtree_str = processed_obs["axtree_txt"]
        human_prompt.append({
            "type": "text",
            "text": f"Step {i}:\nObservation: "
        })
        human_prompt.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{som_screenshot_base64}"
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

def extract_skills(
        website: str,
        traj_path: str,
        model: str = "gpt-4o",
        skill_root_path: str = "src/agentlab/skills",
        id: str = ""
    ):
    try:
        trajectory = get_trajectory_from_annotation(traj_path)
        skills_file_path = f"{skill_root_path}/{website}/skills_{id}.json"
        messages = construct_prompt_messages(website, skills_file_path, trajectory)
        response = generate_from_openai_chat_completion_with_key_pool(messages=messages, model=model, temperature=1.0, max_tokens=2048)
        print("*"*50, "Response during extracting general skills", "*"*50)
        print(response)
        parsed_res_list = parse_html_tag_output(input_string=response, tags=["think", "skill", "steps"])

        # eliminate skills that have been summarized before
        parsed_res_list = [res for res in parsed_res_list if "summarized before" not in res["steps"].lower()]

        # add traj_path to parsed_res_list
        for i, res in enumerate(parsed_res_list):
            res["traj_path"] = traj_path
            res["type"] = "general"
            res["website"] = website
    except Exception as e:
        print(e)
        parsed_res_list = []
    return parsed_res_list

# reset_skills(f"src/agentlab/skills/reddit/skills.json")
# traj_path = f"/home/ubuntu/github/AgentLab/src/agentlab/autogen_policy/annotations/2024-09-01_17-02-43_annotate/2024-09-01_17-02-45_HumanAnnotator_on_webarena.27_51_22d7da"
# print(extract_skills(website="reddit", traj_path=traj_path, model="gpt-4o", skill_root_path="src/agentlab/skills"))
