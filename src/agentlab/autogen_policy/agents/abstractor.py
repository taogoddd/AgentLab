from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_trajectory, get_website_name_from_url, simplify_readable_results
from typing import List, TypedDict, Any
from agentlab.utils.llms import generate_from_4o_chat_completion
import re
import json
import pickle

class Abstractor():

    def construct_prompt_messages(self, skill):
        goal = skill["sub-goal"]
        instruction = skill["instruction"]
        website_name = get_website_name_from_url(skill["trajectory"][0]["obs"]["url"])

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
            return result
        return []
    
    def abstract_skill(self, skills: List[dict]):
        parsed_res_list = []
        for i, skill in enumerate(skills):
            messages = self.construct_prompt_messages(skill)
            res = generate_from_4o_chat_completion(messages)
            parsed_res = self.parse(input_string=res, tags=["generalized sub-goal", "generalized instruction"])
            # TODO: the parsed_res may have a length > 1
            
            # add parsed_res to the skill
            skill["generalized sub-goal"] = parsed_res[0]["generalized sub-goal"]
            skill["generalized instruction"] = parsed_res[0]["generalized instruction"]
            parsed_res_list.append(skill)
        
        return parsed_res_list

    def save_as_readable_json(self, res_list: list[dict], path: str):
        readable_res_list = simplify_readable_results(res_list)
        with open(path, "w") as f:
            json.dump(readable_res_list, f, indent=4)
    
    def save(self, res_list: list[dict], path: str):
        with open(path, "wb") as f:
            pickle.dump(res_list, f)