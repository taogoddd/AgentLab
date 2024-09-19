import gzip
import pickle
from agentlab.agents import dynamic_prompting as dp
from agentlab.utils.llms import generate_from_4o_chat_completion
from PIL import Image
import io
import base64
import os
import re
from openai import AzureOpenAI
import datetime
import json
from dataclasses import dataclass

# TODO: need to modify later
website_name = "Reddit"

# define the url and website name pairs, assuming the port numbers are consistent with those in webarena repo
url_port_website_pairs = {
    "9999": "Reddit",
    "7770": "Shopping Website",
    "7780": "Shopping Admin Website",
    "8023": "Gitlab",
    "8888": "Wikipedia",
    "4399": "Home Page that contains a list of useful links",
}

# TODO: complete these functions
def load_annotation(annotation_dir_path: str):
    return

def construct_file_name_from_ID(ID: int):
    return

def img_array_to_base64(img_array):
    # Convert NumPy array to PIL Image
    img = Image.fromarray(img_array, 'RGB')

    # Save the PIL image to a bytes buffer and encode directly to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64

def get_trajectory_from_annotation(dir_path: str):

    obs_flags=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=True,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    )
    obs_preprocessor = dp.make_obs_preprocessor(obs_flags)

    trajectory = []

    # get the number of steps in the trajectory by counting the number of files matching step_*.pkl.gz
    step_files = [f for f in os.listdir(dir_path) if re.match(r'step_\d+.pkl.gz', f)]
    num_steps = len(step_files)

    # read step info one by one
    for i in range(num_steps):
        file_path = os.path.join(dir_path, f"step_{i}.pkl.gz")
        with gzip.open(file_path, 'rb') as f:
            step_info = pickle.load(f)
            obs = step_info.obs
            action = step_info.action # e.g. click('339')
            reward = step_info.reward

            processed_obs = obs_preprocessor(obs)
            screenshot_array = processed_obs["screenshot"]
            som_screenshot_array = processed_obs["screenshot_som"]

            axtree_str = processed_obs["axtree_txt"]
        
        trajectory.append({
            "obs": obs, # the original observation dict
            "processed_obs": processed_obs, # the processed observation dict
            "action": action, # e.g. click('339')
            "reward": reward,
        })
    
    return trajectory

    # obs: dict_keys(['chat_messages', 'goal', 'open_pages_urls', 'active_page_index', 'url', 'screenshot', 'dom_object', 'axtree_object', 'extra_element_properties', 'focused_element_bid', 'last_action', 'last_action_error', 'last_action_result', 'elapsed_time', 'dom_txt', 'axtree_txt', 'pruned_html', 'screenshot_som'])
    # processed_obs has dom_txt, axtree_txt, pruned_html, screenshot_som

# form the prompt to summarize from human annotations
def construct_summarization_prompt_messages(trajectory):
    goal = trajectory[0]["obs"]["goal"]
    messages = []
    # TODO: format the output better s.t. it can be parsed easily
    system_prompt = """\
You will be given part of the state-action trajectory of a human user interacting with a web page to achieve a specific goal.
Your task is to understand the trajectory and extract an useful new compact action from the trajectory which may contain multiple steps to achieve a sub-goal. You should provided a step-by-step detailed instruction that can help another agent to achieve the sub-goal. If you think the trajectory provided is not enough to extract a complete and useful action, output None for both sub-goal and instruction.
You should follow the format:
<sub-goal>
sub-goal to achieve here
</sub-goal>
<instruction>
instruction to achieve the sub-goal here
</instruction>
<starting-step>
starting step number (starting from 0) of the original trajectory to achieve the sub-goal here
</starting-step>
<ending-step>
ending step number (starting from 0) of the original trajectory to achieve the sub-goal here
</ending-step>
<sub-goal>
another sub-goal
</sub-goal>
<instruction>
instruction to achieve the sub-goal here
</instruction>
<starting-step>
starting step number (starting from 0) of the original trajectory to achieve the sub-goal here
</starting-step>
<ending-step>
ending step number (starting from 0) of the original trajectory to achieve the sub-goal here
</ending-step>
...
Note: 
1. State is the screenshot of the current webpage. In the screenshot, elements are tagged with their IDs. action is the action taken by the user grounding by the element ID.
2. The trajectory provided may be only part of the full trajectory. The full trajectory may contain more steps. 
3. When you generate the subgoal and instructions, you should be aware that this subgoal will be reused as reference when another agent tries to complete a task, so it should be as general as possible.
4. DO NOT include any other words except subgoal and instruction as the format above.
5. Check whether the sub-goal is successfully achieved before you do summarization. If not, do not include it in the summarization.
6. For the sub-goal, it should be a single task and should not contains multiple goals
7. For the instructions, each step should be a single action that can be executed on the webpage.
"""
    prefix = f"""\
Overall goal: {goal}
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
        processed_obs = step["processed_obs"]
        action = step["action"]
        reward = step["reward"]
        screenshot_base64 = img_array_to_base64(processed_obs["screenshot"])
        som_screenshot_base64 = img_array_to_base64(processed_obs["screenshot_som"])
        axtree_str = processed_obs["axtree_txt"]
        human_prompt.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{som_screenshot_base64}"
            }
        })
        human_prompt.append({
            "type": "text",
            "text": f"Action: {action}"
        })
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    messages.append({
        "role": "user",
        "content": human_prompt
    })
    return messages

def construct_abstracting_prompt_messages(experience):
    goal = experience["sub-goal"]
    instruction = experience["instruction"]

    system_prompt = f"""\
You will be given a goal of a task on {website_name} and the corresponding step-by-step instruction to achieve the goal.
Your task is to abstract the goal and instruction to a more general form that can be applied to different cases. You should provide a more general instruction that can be applied to other websites to achieve the same goal.
Here are some examples:
# Example 1
Original sub-goal: 
Comment "WebArena" on post with title "what is the SOTA web navigation agent repo"
Original instruction:
1. Type "SOTA web navigation agent repo" in the search box
2. Click search button
3. Click the post with title "what is the SOTA web navigation agent repo"
4. Click the comment button under the post
5. Type "WebArena" in the comment box
6. Click the submit button
Generalized sub-goal:
Comment {{comment}} on post with title {{post_title}}
Generalized instruction:
1. Type {{post_title}} in the search box
2. Click search button
3. Click the post with title {{post_title}}
4. Click the comment button under the post
5. Type {{comment}} in the comment box
6. Click the submit button
You should follow the format:
<generalized sub-goal>
general sub-goal to achieve here, use {{variable}} to represent the variable part as in the example
</generalized sub-goal>
<generalized instruction>
general instruction to achieve the sub-goal here, use {{variable}} to represent the variable part as in the example
</generalized instruction>

Note:
1. DO NOT include any other words except subgoal and instruction as the format above.
"""
    human_prompt = f"""\
Original sub-goal:
{goal}
Original instruction:
{instruction}
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
def construct_verifying_prompt_messages(experience):
    # TODO: verify the success of the sub-goal through the trajectory
    goal = experience["sub-goal"]
    instruction = experience["instruction"]

    system_prompt = f"""\
You will be given a goal of a task on {website_name} and the corresponding step-by-step instruction to achieve the goal.
Your task is to verify whether the given pair fits the requirements provided.

Requirements:
1. The sub-goal should be a single task and should not contain multiple tasks. For example, "Go to subreddit r/books and upvote the first post" is invalid since it can be split into two sub-goals: "Go to subreddit r/books" and "Upvote the first post".
2. The instruction should be a step-by-step detailed instruction that can help another agent to achieve the sub-goal. Each step should be a single action that can be executed on the webpage.

You should follow the format requirements:
1. if both the sub-goal and instruction are valid, output:
<result>
valid
</result>
<reasons>
None
</reasons>
2. if sub-goal or instruction is invalid, output:
<result>
invalid
</result>
<reasons>
reason for invalidity
</reasons>

for example:
input: 
sub-goal: 
Go to subreddit r/books and upvote the first post
instruction:
1. Type r/books in the search box
2. Click search button
3. Click the first post voting button

output:
<result>
invalid
</result>
<reasons>
The sub-goal contains 2 tasks: "Go to subreddit r/books" and "Upvote the first post"
</reasons>
"""
    human_prompt = f"""\
sub-goal:
{goal}
instruction:
{instruction}
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

def construct_refining_prompt_messages(experience, reasons: str):
    '''
    reasons: reasons for the invalidity of the sub-goal or instruction
    '''
    goal = experience["sub-goal"]
    instruction = experience["instruction"]

    system_prompt = f"""\
You will be given a goal of a task on {website_name} and the corresponding step-by-step instruction to achieve the goal, which has been marked as invalid by another verifier because it does not meet the requirements.
The requirements are:
1. The sub-goal should be a single task and should not contain multiple tasks. For example, "Go to subreddit r/books and upvote the first post" is invalid since it can be split into two sub-goals: "Go to subreddit r/books" and "Upvote the first post".
2. The instruction should be a step-by-step detailed instruction that can help another agent to achieve the sub-goal. Each step should be a single action that can be executed on the webpage.

You will also be provided with the reasons for the invalidity of the sub-goal or instruction. Your task is to refine the sub-goal and instruction to meet the requirements.


"""
    

def propose_action_from_annotation(trajectory):
    messages = construct_summarization_prompt_messages(trajectory)
    res = generate_from_4o_chat_completion(messages, "gpt-4o-2024-05-13")
    return res

# parse the response from the model into a list of results
def parse(input_string: str, tags: list) -> list[dict]:
    if "None" not in input_string:
        # Create a pattern dynamically
        pattern_str = r""
        for tag in tags:
            pattern_str += fr"<{tag}>(.*?)</{tag}>\s*"

        # Compile the pattern
        pattern = re.compile(pattern_str, re.DOTALL)

        # Find all matches
        matches = pattern.findall(input_string)

        # Create a list of dictionaries from the matches
        result = [
            {tag: match[i] for i, tag in enumerate(tags)}
            for match in matches
        ]
        return result
    return []

def get_website_name_from_trajectory(trajectory):
    # traverse the port number and website name pairs to see if the port number is in the trajectory url
    url = trajectory[0]["obs"]["url"]
    for port, website in url_port_website_pairs.items():
        if port in url:
            return website
    return "Unknown"

def abstract_subgoal(experiences: list[dict]):
    res_list = []
    for experience in experiences:
        messages = construct_abstracting_prompt_messages(experience=experience)
        res = generate_from_4o_chat_completion(messages, "gpt-4o-2024-05-13")
        res_list.append(res)
    
    return res_list

def refine_subgoal(experiences: list[dict]):

