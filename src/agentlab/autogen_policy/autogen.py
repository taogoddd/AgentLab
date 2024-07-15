import gzip
import pickle
from agentlab.agents import dynamic_prompting as dp
from PIL import Image
import io
import base64
import os
import re
from openai import AzureOpenAI

client = AzureOpenAI()

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

# now for azure openai
def generate_from_4o_chat_completion(
    messages: list[dict[str, str]],
    model: str,
) -> str:
    if "AZURE_OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "AZURE_OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    
    response = client.chat.completions.create(  # type: ignore
        model=model,
        messages=messages,
    )
    answer: str = response.choices[0].message.content
    return answer

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

def construct_prompt_from_trajectory(trajectory):
    goal = trajectory[0]["obs"]["goal"]
    messages = []
    system_prompt = """\
You will be given part of the state-action trajectory of a human user interacting with a web page to achieve a specific goal.
Your task is to understand the trajectory and extract an useful new compact action from the trajectory which may contain multiple steps to achieve a sub-goal.
Note: 
1. State is the screenshot of the current webpage. In the screenshot, elements are tagged with their IDs. action is the action taken by the user grounding by the element ID.
2. The trajectory provided may be only part of the full trajectory. The full trajectory may contain more steps. 
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

trajectory = get_trajectory_from_annotation("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/annotations/2024-07-15_21-04-39_annotate/2024-07-15_21-04-41_HumanAnnotator_on_webarena.0_51_2433c2")
messages = construct_prompt_from_trajectory(trajectory)
res = generate_from_4o_chat_completion(messages, "gpt-4o-2024-05-13")
print(res)