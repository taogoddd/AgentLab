from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, get_website_name_from_url, simplify_readable_results
from typing import List, TypedDict, Any
from agentlab.utils.llms import generate_from_4o_chat_completion
import re
import json
import pickle

class RLocator():

    def construct_prompt_messages(self, experience):
        goal = experience["sub-goal"]
        trajectory = experience["trajectory"]
        website_name = get_website_name_from_url(experience["trajectory"][0]["obs"]["url"])

        system_prompt = f"""\
    You will be given a goal of a task on {website_name} and the corresponding step-by-step instruction to achieve the goal and also the original trajectory where the goal and instructions are summarized from.
    Your task is to the starting step and the ending step. Assuming the step number starts from 0, starting step is the number of the step, where the sub-goal starts to be attempted in the provided trajectory,
    and the ending step is the number of the step where the sub-goal is achieved in the provided trajectory (inclusive).
    You should follow the format when outputting the starting step and ending step:
    <starting step>
    starting_step_number: int
    </starting step>
    <ending step>
    ending_step_number: int
    </ending step>

    Example output:
    <starting step>
    1
    </starting step>
    <ending step>
    3
    </ending step>
    IMPORTANT NOTES you should absolutely follow:
    1. Refer to the human user trajectory to find the starting step and ending step rather than only looking at the instruction.
    """
        prefix = f"""\
    Sub-goal:
    {goal}
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
                "type": "text",
                "text": f"Step {i}:"
            })
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
    
    def parse(self, input_string: str, tags: list) -> list[dict]:
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

            # post-process the result
            for res in result:
                for key, value in res.items():
                    res[key] = value.strip().replace("\n", "")
                    if res[key].isdigit():
                        res[key] = int(res[key])
                    else:
                        res[key] = "Parsing error"
            return result
        return []

    def locate(self, skills: List[dict]):
        parsed_res_list = []
        for i, skill in enumerate(skills):
            messages = self.construct_prompt_messages(skill)
            res = generate_from_4o_chat_completion(messages)
            parsed_res = self.parse(input_string=res, tags=["starting step", "ending step"])

            skill["starting_step"] = parsed_res[0]["starting step"]
            skill["ending_step"] = parsed_res[0]["ending step"]

            slided_trajectory = skill["trajectory"][int(skill["starting_step"]):int(skill["ending_step"])+1]
            skill["sliced_trajectory"] = slided_trajectory

            parsed_res_list.append(skill)
        
        return parsed_res_list
    
    def save_as_readable_json(self, res_list: list[dict], path: str):
        readable_res_list = simplify_readable_results(res_list=res_list, eleminated_keys=["trajectory", "sliced_trajectory"])
        with open(path, "w") as f:
            json.dump(readable_res_list, f, indent=4)

    def save(self, res_list: list[dict], path: str):
        with open(path, "wb") as f:
            pickle.dump(res_list, f)