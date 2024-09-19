from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_results, get_website_name_from_url
from typing import List, TypedDict, Any
from agentlab.utils.llms import generate_from_4o_chat_completion
import re
import json
import pickle

class Summarizer():

    def construct_prompt_messages(self, trajectory: List[TrajectoryStep]):
        goal = trajectory[0]["obs"]["goal"]
        messages = []
        # TODO: format the output better s.t. it can be parsed easily
        system_prompt = """\
    You will be given part of the state-action trajectory of a human user interacting with a web page to achieve a specific goal.
    Your task is to understand the trajectory and extract an useful new compact action from the trajectory which may contain multiple steps to achieve a sub-goal. You should provided a step-by-step detailed instruction that can help another agent to achieve the sub-goal.
    You should follow the format:
    <sub-goal>
    sub-goal to achieve here
    </sub-goal>
    <instruction>
    instruction to achieve the sub-goal here
    </instruction>
    
    # Example output:
    ## Example 1
    <sub-goal>
    navigate to the subreddit r/books
    </sub-goal>
    <instruction>
    1. Click on the "Forums" tab
    2. Scroll through the list of forums and click on the "Books" subreddit
    </instruction>
    If you think the trajectory provided is not enough to extract a complete and useful action, just output None for all the fields. i.e.:
    <sub-goal>
    None
    </sub-goal>
    <instruction>
    None
    </instruction>
    IMPORTANT NOTES you should absolutely follow: 
    1. State is the screenshot of the current webpage. In the screenshot, elements are tagged with their IDs. action is the action taken by the user grounding by the element ID.
    2. The trajectory provided may be only part of the full trajectory. The full trajectory may contain more steps. 
    3. When you generate the subgoal and instructions, you should be aware that this subgoal will be reused as reference when another agent tries to complete a task, so it should be as general as possible.
    4. DO NOT include any other words except subgoal and instruction as the format above.
    5. Output only one sub-goal and instruction pair.
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
    
    def parse(self, input_string: str, tags: List[str] = ["sub-goal", "instruction"]) -> List[dict]:
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
    
    def summarize(self, trajectory: List[TrajectoryStep]) -> List[dict]:
        messages = self.construct_prompt_messages(trajectory)
        res = generate_from_4o_chat_completion(messages=messages, model="gpt-4o-2024-05-13")
        print(res)
        print("====================================")
        parsed_res_list = self.parse(input_string=res, tags=["sub-goal", "instruction"])
        for r in parsed_res_list:
            r["trajectory"] = trajectory
        # TODO: add a seperate agent to locate the starting and ending step here
        # for r in parsed_res_list:
        #     starting_step = int(r["starting-step"])
        #     ending_step = int(r["ending-step"])
        #     sliced_window = trajectory[starting_step:ending_step+1]
        #     r["trajectory"] = sliced_window
        return parsed_res_list
        
    def sliding_summarize(self, trajectory: List[TrajectoryStep], window_size: int):
        res_list = []
        for i in range(0, len(trajectory) - window_size + 1):
            window = trajectory[i:i+window_size]
            parsed_res_list = self.summarize(window)
            res_list.extend(parsed_res_list)
        return res_list
    
    def save_as_readable_json(self, res_list: list[dict], path: str):
        readable_res_list = simplify_readable_results(res_list)
        with open(path, "w") as f:
            json.dump(readable_res_list, f, indent=4)
    
    def save(self, res_list: list[dict], path: str):
        with open(path, "wb") as f:
            pickle.dump(res_list, f)
