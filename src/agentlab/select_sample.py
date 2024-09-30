from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_trajectory, get_trajectory_from_annotation, get_website_name_from_url
import os
from webarena.llms.providers.openai_utils import full_generate_from_openai_chat_completion_with_key_pool
import re
import numpy as np
from browsergym.experiments.utils import count_tokens, count_messages_token

def count_multimodal_messages_tokens(messages, model="gpt-4") -> int:
    token_count = 0
    for message in messages:
        if hasattr(message, "content"):
            message = message.content

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

# for a 1500*1280 image, 765 each high detailed image and 85 each low detailed image
def truncate_trajectory_by_tokens(trajectory: list, max_tokens: int = 128000-20000, model="gpt-4") -> list:
    truncated_trajectory = []
    token_count = 0
    # from back to front
    for i, step in enumerate(trajectory[::-1]):
        token_count += count_tokens(step["think"], model)
        token_count += count_tokens(step["action"], model)
        # check whether this is the last 10 images
        if i < 10:
            token_count += 765
        else:
            token_count += 85
        if token_count > max_tokens:
            break
        truncated_trajectory.append(step)
    return truncated_trajectory[::-1]


def proc_trajectory(trajectory: list) -> list:
    return [step for step in trajectory if step["action"] != None]

def format_think_action_str(trajectory: list) -> str:
    think_action_str = ""
    for i, step in enumerate(trajectory):
        think_action_str += f"Step {i}:\n"
        think_action_str += "Think: " + step["think"] + "\n"
        think_action_str += "Action: " + step["action"] + "\n"
    return think_action_str

def evaluate_success(trajectory, models: list[str] = ["gpt-4o" for _ in range(4)], n: int = 20, top_p: float = 1.0, should_log: bool = False) -> float:
    trajectory = truncate_trajectory_by_tokens(trajectory)

    current_url = trajectory[0]["obs"]["url"]
    intent = trajectory[0]["obs"]["goal"]
    last_think_action_str = format_think_action_str(trajectory)

    content = []

    for i, step in enumerate(trajectory):
        obs = step["obs"]
        processed_obs = step["processed_obs"]
        action = step["action"]
        # reward = step["reward"]
        # screenshot_base64 = img_array_to_base64(processed_obs["screenshot"])
        som_screenshot_base64 = img_array_to_base64(processed_obs["screenshot_som"])
        # axtree_str = processed_obs["axtree_txt"]
        # content.append({
        #     "type": "text",
        #     "text": f"Step {i}:\nObservation: "
        # })
        # check whether the image is the last k images
        k = 10
        flag = False
        if i >= len(trajectory) - k:
            flag = True
        if flag:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{som_screenshot_base64}",
                    "detail": "high"
                }
            })
        else:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{som_screenshot_base64}",
                }
            })
        # content.append({
        #     "type": "text",
        #     "text": f"URL: {obs['url']}"
        # })
        # content.append({
        #     "type": "text",
        #     "text": f"Action: {action}"
        # })
        content.append({
            "type": "text",
            "text": f"""User Intent: {intent}
Think and Action History: {last_think_action_str}
Bot response to the user: {trajectory[-1]["action"]}
Current URL: {current_url}
The last {len(trajectory)} snapshots of the agent's trajectory are shown in the {len(trajectory)} images. The LAST IMAGE represents the current state of the webpage.
"""
        })
        # 650 tokens of system messages
        messages = [
        {
            "role": "system",
            "content": f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
"""
        },
        {
            "role": "user",
            "content": content
        }
    ]
    all_responses = []
    for model in models:
        response = full_generate_from_openai_chat_completion_with_key_pool(
            model=model,
            messages=messages,
            max_tokens=256,
            top_p=top_p,
            n=n // len(models)
        )
        all_responses.extend(response.choices)

    if should_log:
        print('=' * 30)
        print("Value function input:", content[-1])
    all_scores = []
    for r_idx, r in enumerate(all_responses):
        if should_log:
            print(f"Output {r_idx}: {r.message.content}")
        try:
            pred = re.search(r'Status: "?(.+)"?', r.message.content).group(1)
            if 'success' in pred.lower():
                score = 1.0
            else:
                # Check if it's on the path to success
                on_path = re.search(r'On the right track to success: "?(.+)"?', r.message.content).group(1)
                if 'yes' in on_path.lower():
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            score = 0.0
        
        all_scores.append(score)
    
    score = np.mean(all_scores)
    if should_log:
        print(f"Final score: {score}")
        print('=' * 30)
    return score

def select_best_sample(result_parent_dir: str, sample_ids: list[int], max_steps: int = 30) -> int:
    scores = []
    for sample_id in sample_ids:
        task_traj_dir = os.path.join(result_parent_dir, str(sample_id))
        trajectory = get_trajectory_from_annotation(task_traj_dir)
        trajectory = proc_trajectory(trajectory)
        scores.append(evaluate_success(trajectory))
    best_sample_id = sample_ids[scores.index(max(scores))]
    return best_sample_id