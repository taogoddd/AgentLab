from typing import Any, List, TypedDict
import numpy as np
import gzip
import pickle
from agentlab.agents import dynamic_prompting as dp
from PIL import Image
import io
import base64
import os
import re
from openai import AzureOpenAI
import datetime
import json
from dataclasses import dataclass

class Obs(TypedDict):
    chat_messages: List
    goal: str
    open_pages_urls: List[str]
    active_page_index: np.ndarray
    url: str
    screenshot: np.ndarray
    dom_object: Any
    axtree_object: Any
    extra_element_properties: Any
    focused_element_bid: str
    last_action: str
    last_action_error: str
    last_action_result: str
    elapsed_time: Any
    dom_txt: str
    axtree_txt: str
    pruned_html: str
    screenshot_som: np.ndarray

class ProcessedObs(TypedDict):
    dom_txt: str
    axtree_txt: str
    pruned_html: str
    screenshot_som: np.ndarray

class TrajectoryStep(TypedDict):
    obs: Obs
    processed_obs: ProcessedObs
    action: str
    reward: int

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

def img_array_to_base64(img_array):
    # Convert NumPy array to PIL Image
    img = Image.fromarray(img_array, 'RGB')

    # Save the PIL image to a bytes buffer and encode directly to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64

def simplify_readable_trajectory(trajectory: List[TrajectoryStep]):
    '''
    only keey necessary information in the trajectory step
    '''
    simplified_trajectory = []
    for step in trajectory:
        simplified_step = {
            "obs": {
                "url": step["obs"]["url"],
            },
            "processed_obs": {
                "axtree_txt": step["processed_obs"]["axtree_txt"]
            },
            "action": step["action"],
            "reward": step["reward"]
        }
        simplified_trajectory.append(simplified_step)
    return simplified_trajectory

def simplify_readable_results(res_list: List[dict], eleminated_keys: List[str] = ["trajectory"]) -> List[dict]:
    '''
    pop out unjsonifiable keys in the trajectory part of the res_list
    '''
    simplified_res_list = []
    for res in res_list:
        simplified_res = res.copy()
        for key in eleminated_keys:
            simplified_res.pop(key, None)
        simplified_res_list.append(simplified_res)
    return simplified_res_list

def get_website_name_from_url(url: str):
    # define the url and website name pairs, assuming the port numbers are consistent with those in webarena repo

    # the names are consistent with the config files
    url_port_website_pairs = {
        "9999": "reddit",
        "7770": "shopping",
        "7780": "shopping_admin",
        "8023": "gitlab",
        "8888": "wiki",
        "4399": "Home Page that contains a list of useful links",
    }
    # get the port number from the url e.g. http://localhost:9999/f/news -> 9999
    port = url.split(":")[-1].split("/")[0]
    website_name = url_port_website_pairs.get(port, "Unknown Website")
    return website_name