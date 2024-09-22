from typing import List
import json
from agentlab.utils.utils import parse_html_tag_output, get_website_description
from webarena.llms.providers.openai_utils import generate_from_openai_chat_completion_with_key_pool
import re

def parse_selection_output(input_string: str) -> List[str]:
    try:
        # Regular expression to match 'id' and 'name' pairs
        pattern = r'id: (\d+); name: ([^;]+)'

        # Find all matches in the input string
        matches = re.findall(pattern, input_string)

        # Convert matches to a list of dictionaries
        parsed_list = [{'id': int(match[0]), 'name': match[1].strip()} for match in matches]
    except Exception as e:
        print(f"Error: {e}")
        parsed_list = []
    return parsed_list

def select_navi_skills(intent: str, website: str, navi_skills, model: str, max_skills: int = 5):
    """
    Select the navigation skills based on the intent and website.

    Args:
    - intent: The intent of the task.
    - website: The website the task is on.
    - navi_skills: The navigation skills to select from.

    Returns:
    - A dictionary of navigation skills based on the intent and website.
    """

    def construct_prompt_messages(intent: str, website: str, navi_skills, max_skills: int = 5):
        navi_skill_str = ""
        for i, skill in enumerate(navi_skills):
            URL = skill["URL"]
            name = skill["name"]
            description = skill["description"]
            usages = skill["usages"]
            navi_skill_str += f"id: {i + 1}; name: {name}; description: {description}; possible usages: {usages}; url: {URL}\n"
        system_prompt = f"""\
You will be given a goal of a task to be executed on a website and a list of urls and the corresponding page summary to choose from.
You need to select the pages that most possibly need to be visited to achieve the goal.
You should break the task down into a few steps so that you can select the pages that can help most in each step.
IMPORTANT: You should select not more than {max_skills} pages!

Output format:
<think>
think step by step. Break the task down into a few steps and then select pages
</think>
<selected-pages>
id: {{the id number (the number at the beginning) of page 1}}; name: page 1 name
id: {{the id number (the number at the beginning) of page 2}}; name: page 2 name
...
</selected-pages>

# Examples
## Example 1

Task goal: Upvote the hottest post in r/books
Current website: {get_website_description(website)}
Shortcuts to choose from:
id: 1; name: Forums page; description: It shows a list of different forums; possible usages: navigate to different forums; url: https://www.example.com/forums
id: 2; name: Profile page; description: It shows the information of current user; possible usages: Check or modify the information of the current user; url: https://www.example.com/profile
id: 3; name: Submission page; description: It provides a few text boxes to fill in to submit a new post; possible usages: Submit new posts; url: https://www.example.com/submission
id: 4: name: Subscribed forums page; description: It provides a list of subscribed forums; possible usages: check or navigate to subscribed forums; url: https://www.example.com/subscribed

Output:
<think>
The goal is to upvote the hottest post in r/books. The user needs to navigate to the r/books page first or go to forums to find the r/books page. Then the user needs to find the hottest post in the r/books page. So the useful pages from the shortcuts are Forums page
</think>
<selected-pages>
id: 1; name: Forums page
</selected-pages>"""
        human_prompt = f"""\
Task goal: {intent}
Current website: {get_website_description(website)}
Shortcuts to choose from:
{navi_skill_str}
"""
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
    
    messages = construct_prompt_messages(intent, website, navi_skills)
    response = generate_from_openai_chat_completion_with_key_pool(messages=messages, model=model, temperature=0.7, max_tokens=2048)
    print("*"*50, "Response during select_navi_skills", "*"*50)
    print(response)
    parsed_res_list = parse_html_tag_output(input_string=response, tags=["think", "selected-pages"])
    if len(parsed_res_list) == 1:
        parsed_res = parsed_res_list[0]
        selected_pages = parse_selection_output(parsed_res["selected-pages"])
        selected_navis = [navi_skills[int(page["id"]) - 1] for page in selected_pages]
    else:
        selected_navis = []

    # slice if more than max_skills
    if len(selected_navis) > max_skills:
        selected_navis = selected_navis[:max_skills]
    return selected_navis

def select_general_skills(intent: str, website: str, general_skills, model, max_skills: int = 5) -> dict[str, List[str]]:
    """
    Select the general skills based on the intent and website.

    Args:
    - intent: The intent of the task.
    - website: The website the task is on.
    - general_skills: The general skills to select from.

    Returns:
    - A dictionary of general skills based on the intent and website.
    """

    def construct_prompt_messages(intent: str, website: str, general_skills, max_skills: int = 5):
        general_skill_str = ""
        for i, skill in enumerate(general_skills):
            skill_name = skill["skill"]
            steps = skill["steps"]
            general_skill_str += f"Skill {i + 1}: {skill_name}\n"
            general_skill_str += f"{steps}\n"
        system_prompt = f"""\
You will be given a goal of a task to be executed on a website and a list of skills to choose from.
You need to select the skills that can help most in achieving the goal.

You should break the task down into a few steps so that you can select the skills that can help most in each step.
IMPORTANT: You should select not more than {max_skills} skills!

Output format:
<think>
think step by step. Break the task down into a few steps and then select skills
</think>
<selected-skills>
id: {{the id number (the number at the beginning) of skill 1}}; name: skill 1 name
id: {{the id number (the number at the beginning) of skill 2}}; name: skill 2 name
...
</selected-skills>

# Examples
## Example 1

Task goal: Upvote the hottest post in r/books
Current website: {get_website_description("reddit")}
Skills to choose from:
Skill 1: Navigate to forums
1. Click on the "Forums" menu item.
```click({{forums id}})```
2. Click on the specific forum name.
```click({{forum name id}})```
Skill 2: Submit a new post
1. Type the post title in the title text box.
```type({{title text box id}}, "Post Title")```
2. Type the post content in the content text box.
```type({{content text box id}}, "Post Content")```
3. Click on the "Submit" button.`
``click({{submit button id}})```
Skill 3: Sort posts by {{sort criterion}}
1. Click on the "Sort by" dropdown menu.
```click({{sort by dropdown id}})```
2. Select the {{sort criterion}} option from the "Sort by" dropdown menu.
```click({{sort criterion id}})```

Output:
<think>
The goal is to upvote the hottest post in r/books. The user needs to navigate to the r/books page first or go to forums to find the r/books page. Then the user needs to find the hottest post in the r/books page. So the useful skills from the shortcuts are Navigate to forums, Sort posts by hotness
</think>
<selected-skills>
id: 1; name: Navigate to forums
id: 3; name: Sort posts by hotness
</selected-skills>

Notes:
1. Some skills might not be consistent with the current task but it is still useful to refer to, e.g. write a post to express happiness is useful in a task to write a post to express sadness.
"""
        human_prompt = f"""\
Task goal: {intent}
Current website: {get_website_description(website)}
Skills to choose from:
{general_skill_str}"""
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
    messages = construct_prompt_messages(intent, website, general_skills)
    response = generate_from_openai_chat_completion_with_key_pool(messages=messages, model=model, temperature=0.7, max_tokens=2048)
    print("*"*50, "Response during select_general_skills", "*"*50)
    print(response)
    parsed_res_list = parse_html_tag_output(input_string=response, tags=["think", "selected-skills"])
    if len(parsed_res_list) == 1:
        parsed_res = parsed_res_list[0]
    else:
        parsed_res = {"selected-skills": ""}
    selected_skills = parse_selection_output(parsed_res["selected-skills"])
    selected_generals = [general_skills[int(skill["id"]) - 1] for skill in selected_skills]

    # slice if more than max_skills
    if len(selected_generals) > max_skills:
        selected_generals = selected_generals[:max_skills]
    return selected_generals

def select_skills(intent: str, website: str, skill_path: str, model: str = "gpt-4o", max_skills = (5, 5)) -> dict[str, List[str]]:
    """
    Select the skills based on the intent and website.

    Args:
    - intent: The intent of the task.
    - website: The website the task is on.
    - skill_path: The path to the skill file to load for the agent.

    Returns:
    - A dictionary of skills based on the intent and website.
    """

    with open(skill_path, "r") as f:
        skills = json.load(f)

    max_navi_skills, max_general_skills = max_skills
    
    navi_skills = [skill for skill in skills if skill["type"] == "navi"]
    selected_navi_skills = select_navi_skills(intent, website, navi_skills, model, max_navi_skills)
    print("*"*50, "Selected navigation skills", "*"*50)
    print(selected_navi_skills)

    general_skills = [skill for skill in skills if skill["type"] == "general"]
    selected_general_skills = select_general_skills(intent, website, general_skills, model, max_general_skills)
    print("*"*50, "Selected general skills", "*"*50)
    print(selected_general_skills)

    return selected_navi_skills + selected_general_skills