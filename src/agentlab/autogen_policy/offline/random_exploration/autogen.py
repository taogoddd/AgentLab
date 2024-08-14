import gzip
import pickle
from agentlab.agents import dynamic_prompting as dp
from agentlab.utils.llms import generate_from_4o_chat_completion
from PIL import Image
import io
import base64
import os
import re
import json
from openai import AzureOpenAI
from tqdm import tqdm

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

def get_trajectory_from_exploration(dir_path: str):

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

# parse the response from the model
def parse(input_string):
    # Define the regular expression pattern
    pattern = re.compile(
        r"<sub-goal>(.*?)</sub-goal>\s*<starting-step>(.*?)</starting-step>\s*<ending-step>(.*?)</ending-step>\s*<instruction>(.*?)</instruction>",
        re.DOTALL
    )

    # Find all matches
    matches = pattern.findall(input_string)

    # Create a list of dictionaries from the matches
    result = [
        {
            "sub_goal": match[0].strip(),
            "starting_step": match[1].strip(),
            "ending_step": match[2].strip(),
            "instruction": match[3].strip(),
        }
        for match in matches
    ]
    
    return result

def propose_action_from_exploration(trajectory):
    messages = []
    # TODO: format the output better s.t. it can be parsed easily
    system_prompt = """\
You will be given part of the state-action trajectory of a human user interacting with a web page to explore a website.
Your task is to understand the trajectory and extract an useful new compact action from the trajectory which may contain multiple steps to achieve a sub-goal.
You should follow the format, if you have multiple sub-goals, just repeat the format:
<sub-goal>
sub-goal to achieve here
</sub-goal>
<starting-step>
starting step number (starting from 0) of the original trajectory to achieve the sub-goal here
</starting-step>
<ending-step>
ending step number (starting from 0) of the original trajectory to achieve the sub-goal here
</ending-step>
<instruction>
summarized instruction to achieve the sub-goal here for further reference
</instruction>
...
Note:
1. State is the screenshot of the current webpage. In the screenshot, elements are tagged with their IDs. action is the action taken by the user grounding by the element ID.
2. The trajectory provided may be only part of the full trajectory. The full trajectory may contain more steps. 
3. When you generate the subgoal and instructions, you should be aware that this subgoal will be reused as reference when another agent tries to complete a task, so it should be as general as possible.
4. DO NOT include any other words except subgoal and instruction as the format above.
5. Check whether the sub-goal is successfully achieved before you do summarization. If not, do not include it in the summarization.
"""
    prefix = f"""\
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
    res = generate_from_4o_chat_completion(messages, "gpt-4o-2024-05-13")
    return res

def get_experience_from_exploration(path: str, window_size: int, step_size: int):
    trajectory = get_trajectory_from_exploration(path)
    experience = []
    # do as a sliding window, with step size of 1
    for i in tqdm(range(0, len(trajectory) - window_size + 1, step_size)):
        window = trajectory[i:i+window_size]
        res = propose_action_from_exploration(window)
        parsed_res = parse(res)

        starting_step = i + int(parsed_res[0]["starting_step"])
        ending_step = i + int(parsed_res[0]["ending_step"])
        sub_goal = parsed_res[0]["sub_goal"]
        instruction = parsed_res[0]["instruction"]
        obs = trajectory[starting_step]["obs"]

        experience.append({
            "sub_goal": sub_goal,
            "obs": obs,
            "instruction": instruction,
            "starting_step": starting_step,
            "ending_step": ending_step,
        })
    
    return experience

def summarize_state(path: str, obs_type: str, processed_obs: dict):
    if obs_type == "axtree":
        axtree = processed_obs["axtree_txt"]
        # TODO: modify the prompt
        system_prompt = f"""\
You will be given the accessibility tree (AXTree) of a web page.
Your task is to summarize the content of the webpage through its AXTree.
Directly output the summary"""
        human_prompt = f"""\
AXTree:
{axtree}
"""
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": axtree
            }
        ]
        res = generate_from_4o_chat_completion(messages, "gpt-4o-2024-05-13")
        return res
    elif obs_type == "screenshot":
        screenshot_base64 = img_array_to_base64(processed_obs["screenshot"])
        # TODO
        system_prompt = f"""\
You will be given the screenshot of a web page.
Your task is to summarize the content of the webpage through its screenshot.
Directly output the summary of the screenshot."""
        human_prompt = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{screenshot_base64}"
                }
            }
        ]
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
        res = generate_from_4o_chat_completion(messages, "gpt-4o-2024-05-13")
        return res
    else:
        raise ValueError("Invalid obs_type")

def state_summaries_from_exploration(path: str, obs_type: str):
    """
    Generate state summaries from the exploration trajectory 

    obs_type: "html", "axtree", "screenshot", "screenshot_som"
    """
    messages = []
    trajectory = get_trajectory_from_exploration(path)
    state_summaries = []
    for i, step in tqdm(enumerate(trajectory)):
        obs = step["obs"]
        processed_obs = step["processed_obs"]
        res = summarize_state(path, obs_type, processed_obs)
        state_summaries.append(
            {
                "obs": obs,
                "summary": res
            }
        )
    return state_summaries 

path = "/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/explorations/2024-08-11_23-09-06_explore/2024-08-11_23-09-07_Explorer_on_webarena.0_51_671048"
window_size = 10
step_size = 5
experience = get_experience_from_exploration(path, window_size, step_size)

# rewriting mode
with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences/experiences.pkl", "wb") as f:
    pickle.dump(experience, f)

# write to json file to visualize

# only keep url and axtree_txt in obs
for exp in experience:
    exp["obs"] = {
        "url": exp["obs"]["url"],
        "axtree_txt": exp["obs"]["axtree_txt"]
    }

with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences_json/experiences.json", "w") as f:
    json.dump(experience, f, indent=4)

# TODO: appending mode

# messages = construct_prompt_from_trajectory(trajectory)
# res = generate_from_4o_chat_completion(messages, "gpt-4o-2024-05-13")
# print(res)