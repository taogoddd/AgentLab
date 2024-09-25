from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_trajectory, get_trajectory_from_annotation, get_website_name_from_url
from webarena.llms.providers.openai_utils import generate_from_openai_chat_completion_with_key_pool
from agentlab.utils.utils import parse_html_tag_output, get_website_description
import re
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

EXCLUDED_URLS = {
    "shopping_admin": [],
    "shopping": [],
    "reddit": [],
    "gitlab": [],
    "map": [],
    "classifieds": [],
}
website_base_depth = {
    "shopping_admin": 3,
    "shopping": 2,
    "reddit": 2,
    "gitlab": 2,
    "map": 2,
    "classifieds": 2,
}
website_max_depth = {
    "shopping_admin": 4,
    "shopping": 3,
    "reddit": 3,
    "gitlab": 3,
    "map": 3,
    "classifieds": 3,
}
def eval_URL(url: str, website: str) -> bool:
    if url in EXCLUDED_URLS[website]:
        return False
    if "dashboard" in url:
        return False
    # check if the url has too many levels
    # remove the first part
    url = url.split("://")[-1]
    # remove ending /
    url = url.rstrip("/")
    if len(url.split("/")) > website_max_depth[website]:
        return False
    return True

def parse_page_summary(page_summary: str):
    # Modified pattern to handle optional new lines and flexible spacing
    pattern = r"Name:\s*(.*?)\s*Description:\s*(.*?)\s*Usages:\s*(.*?)\s*$"

    match = re.search(pattern, page_summary, re.DOTALL)  # re.DOTALL allows '.' to match newlines as well
    if match:
        name = match.group(1).strip()
        description = match.group(2).strip()
        usages = match.group(3).strip()
        return name, description, usages
    return None, None, None

# this desc str is for extracting skills from a trajectory only
def get_skills_desc(skills_path: str):
    # load skills
    with open(skills_path, "r") as f:
        skills = json.load(f)
    if not skills:
        return ["No skills learned yet."]
    skills_str = "Existing summarized pages:\n"
    for i, skill in enumerate(skills):
        if skill["type"] == "navi":
            skills_str += f"Page {i+1}: {skill['name']}\n"
            skills_str += f"Description: {skill['description']}\n"
            skills_str += f"Usages: {skill['usages']}\n"
            skills_str += f"URL: {skill['URL']}\n"
    
    return skills_str

def construct_prompt_messages(
        website: str,
        skill_file_path: str,
        trajectory: list[TrajectoryStep],
    ):
    existing_pages_str = get_skills_desc(skill_file_path)
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
You need to summarize the useful pages and pair it up with the corresponding URLs.

Output format:
<URL>
the URL of page 1
</URL>
<think>
think step by step like in the examples and summarize the page
</think>
<page-summary>
the brief summary of page 1, following the format:
Name: '{{name}} page
Description: {{descriptions}}
Usages: {{usages}}'
</page-summary>
<URL>
the URL of page 2
</URL>
<think>
think step by step like in the examples and summarize the page
</think>
<page-summary>
the brief summary of page 2, following the format:
Name: '{{name}} page
Description: {{descriptions}}
Usages: {{usages}}'
</page-summary>
...

# Examples
## Example 1
Overall goal of the trajectory: Go to r/books forum.
Current website: Reddit
Existing summarized pages:
Page 1: Profile page
Description: it shows the user's profile information.
Usages: view or modify user's profile information.
URL: https://www.example.com/profile
Human user trajectory: [neglected here]

Output:
<URL>
https://www.example.com/forums
</URL>
<think>
From the content of the page, it shows a list of forums, I can summarize it to Forum page. The website is Reddit so this page can be used to navigate to different forums.
</think>
<page-summary>
Name: Forums page; Description: it shows a list of different forums.; Possible usages: navigate to different forums.
</page-summary>

IMPORTANT NOTES you should absolutely follow: 
1. DO NOT include any other words except url, think and page summary as the format stated above.
2. Follow the example to think and summarize the page.
3. You should only summarize once for each unique URL.
4. Check existing pages before generating, do not summarize pages that have already been summarized, instead, use "Summarized before" in the steps.
5. Focus on the main content of the page and may ignore the modifications made by the user when generating the summary.
"""
#     prefix = f"""\
# Overall goal of the trajectory: {goal}
# Current website: {get_website_description(website)}
# Existing summarized pages:
# {existing_pages_str}
# Human user trajectory:
# """
    goal = f"""\
Overall goal of the trajectory: {goal}
Images relevant to the goal:
"""
    other_info = f"""\
Current website: {get_website_description(website)}
Existing summarized pages:
{existing_pages_str}
Human user trajectory:
"""
    human_prompt = [
        {
            "type": "text",
            "text": goal
        },
    ]
    for i, img_url in enumerate(img_urls, start=1):
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

def extract_navi_skill(
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
        print("*"*50, "Response during extracting dynamics", "*"*50)
        print(response)
        parsed_res_list = parse_html_tag_output(input_string=response, tags=["URL", "think", "page-summary"])
        # check if the URL has been extracted
        # load skills
        with open(f"{skill_root_path}/{website}/skills_{id}.json", "r") as f:
            skills = json.load(f)
        parsed_res_list = [res for res in parsed_res_list if not any(skill["URL"] == res["URL"] for skill in skills if skill["type"] == "navi") and eval_URL(res["URL"], website)]
        # add traj_path to parsed_res_list
        for i, res in enumerate(parsed_res_list):
            summary = res["page-summary"]
            name, description, usages = parse_page_summary(summary)
            res["name"] = name
            res["description"] = description
            res["usages"] = usages
            res["traj_path"] = traj_path
            res["website"] = website
            res["type"] = "navi"
    except Exception as e:
        parsed_res_list = []
        raise e
        
    return parsed_res_list

# traj_path = f"/home/ubuntu/agentlab_results/2024-09-11_02-01-51_baseline/2024-09-11_02-01-52_GenericAgent_on_webarena.13_2_ab01c0"
# print(extract_navi_skill(website="reddit", traj_path=traj_path, model="gpt-4o"))
