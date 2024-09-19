from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_results, get_website_name_from_url
from typing import List, TypedDict, Any
from agentlab.utils.llms import generate_from_4o_chat_completion
import re
import json
import pickle

class Slicer():

    def construct_prompt_messages(self, trajectory: List[TrajectoryStep]):
        goal = trajectory[0]["obs"]["goal"]
        messages = []
        # TODO: format the output better s.t. it can be parsed easily
        system_prompt = f"""\
    You will be given part of the state-action trajectory of a human user interacting with a web page, the overall goal of the trajectory and a list atomic actions that the original user can take.
    Your task is to:
    decompose the trajectory into several high-level sub-goals.
    
    Sub-goal here is a high-level step that the user takes to achieve the overall goal, and a subgoal should take some elementary actions to achieve.
    Overall goal: {goal}

    When outputing, you should follow the format:
    <think>
    think step by step and check the requirements of the subgoal
    <think>
    <sub-goal>
    sub-goal 1 to achieve here
    </sub-goal>
    <think>
    think step by step and check the requirements of the subgoal
    <think>
    <sub-goal>
    sub-goal 2 to achieve here
    </sub-goal>
    ...
    
    # Example output:
    ## Example 1
    Overall goal: Upvote the post titled "What is the best book you have ever read?"
    
    <think>
    From the trajectory, the user first searched for the post titled "What is the best book you have ever read?". Also, this sub-goal contains atomic actions: fill, click.
    As for type, the user is interacting with the current page. So the type of this sub-goal is interacting.
    </think>
    <sub-goal>
    search for the post titled "What is the best book you have ever read?"
    </sub-goal>
    <think>
    From the trajectory, the user then clicked on the most relevant post. Also, this sub-goal contains atomic actions: click.
    </think>
    <sub-goal>
    click on the most relevant post
    </sub-goal>
    <think>
    From the trajectory, the user then upvoted the post. Also, this sub-goal contains atomic actions: click.
    </think>
    <sub-goal>
    upvote the post
    </sub-goal>
    
    IMPORTANT NOTES you should absolutely follow: 
    1. Each subgoal should take some elementary actions to achieve. Otherwise, the subgoal is not valid and you should not consider it as a subgoal, e.g. identify / count sth.
    2. State is the screenshot of the current webpage. In the screenshot, elements are tagged with their IDs. action is the action taken by the user grounding by the element ID.
    3. DO NOT include any other words except subgoals as the format stated above.
    4. For the sub-goals, each one should be a single task and should not contains multiple goals.
    5. There might be some noise in the trajectory, if some part of the trajectory is not useful for the sub-goal, you can ignore it.
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
    
    def parse(self, input_string: str, tags: List[str] = ["think", "sub-goal"]) -> dict[str, List[str]]:
        # TODO: add error handling
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
    
    def slice(self, trajectory: List[TrajectoryStep]) -> List[dict]:
        messages = self.construct_prompt_messages(trajectory)
        res = generate_from_4o_chat_completion(messages=messages, model="gpt-4o-2024-05-13")
        print(res)
        print("====================================")
        parsed_res_list = self.parse(input_string=res)
        for r in parsed_res_list:
            r["trajectory"] = trajectory
        # TODO: add a seperate agent to locate the starting and ending step here
        # for r in parsed_res_list:
        #     starting_step = int(r["starting-step"])
        #     ending_step = int(r["ending-step"])
        #     sliced_window = trajectory[starting_step:ending_step+1]
        #     r["trajectory"] = sliced_window
        return parsed_res_list # list here for formatting
    
    def save_as_readable_json(self, res_list: list[dict], path: str):
        readable_res_list = simplify_readable_results(res_list)
        with open(path, "w") as f:
            json.dump(readable_res_list, f, indent=4)
    
    def save(self, res_list: list[dict], path: str):
        with open(path, "wb") as f:
            pickle.dump(res_list, f)