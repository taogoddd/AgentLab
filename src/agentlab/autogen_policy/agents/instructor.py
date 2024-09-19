from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_results, get_website_name_from_url
from typing import List, TypedDict, Any
from agentlab.utils.llms import generate_from_4o_chat_completion
import re
import json
import pickle

class NLInstructor():

    def construct_prompt_messages(self, sliced_skill: dict):
        goal = sliced_skill["sub-goal"]
        trajectory = sliced_skill["trajectory"]
        messages = []
        # TODO: format the output better s.t. it can be parsed easily
        system_prompt = f"""\
    You will be given the state-action trajectory of a human user interacting with a web page, a sub-goal that summarizes a part of the trajectory, and a list atomic actions that the original user can take.
    Your task is to:
    Given the sub-goal, provide a step-by-step detailed instruction based on the trajectory for the sub-goal that can help another agent to achieve the sub-goal. 

    Sub-goal: {goal}

    Atomic actions: 
    11 different types of actions are available.

        noop(wait_ms: float = 1000)
            Description: Do nothing, and optionally wait for the given time (in milliseconds).

        send_msg_to_user(text: str)
            Description: Sends a message to the user.

        scroll(delta_x: float, delta_y: float)
            Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.

        fill(bid: str, value: str)
            Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.

        select_option(bid: str, options: str | list[str])
            Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.

        click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
            Description: Click an element.

        clear(bid: str)
            Description: Clear the input field.

        hover(bid: str)
            Description: Hover over an element.

        go_back()
            Description: Navigate to the previous page in history.

        go_forward()
            Description: Navigate to the next page in history.

        goto(url: str)
            Description: Navigate to a url.

    When outputing, you should follow the format:
    <think>
    think step-by-step, you should firstly locate certain part of trajectory which aims to achieve the sub-goal and then summarize them into a detailed instruction.
    <think>
    <instruction>
    step-by-step detailed instruction to achieve the sub-goal here
    </instruction>
    
    # Example output:
    ## Example 1
    Sub-goal: navigate to the post titled "What is the best book you have ever read?"

    <think>
    This sub-goal 'navigate to the post titled "What is the best book you have ever read?"' is attempted from step 0 to step 2. The user firstly fills out the search bar with the text "What is the best book you have ever read?" and then clicks on the search button. Finally, the user clicks on the most relevant post to navigate to the post.
    </think>
    <instruction>
    1. Fill out the search bar with the text "What is the best book you have ever read?"
    2. Click on the search button
    3. Click on the most relevant post to navigate to the post
    </instruction>
    
    IMPORTANT NOTES you should absolutely follow: 
    1. State is the screenshot of the current webpage. In the screenshot, elements are tagged with their IDs. action is the action taken by the user grounding by the element ID.
    2. DO NOT include any other words except the instruction as the format stated above.
    3. For the instruction, each step should be able to be executed with atomic actions above. Counter examples are like "Identify sth" or "Find sth" because they are not executable with the atomic actions above.
    4. Do not include the exact IDs of the elements in the instruction. You should use the element type or the text of the element to refer to the element.
    5. There could be some noise in the trajectory for the sub-goal. For example, there could be some irrelevant actions that will not contribute to the sub-goal. You should filter out these irrelevant actions when generating the instruction.
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
    
    def parse(self, input_string: str, tags: List[str] = ["instruction"]) -> dict[str, List[str]]:
        # TODO: add error handling

        # strip
        input_string = input_string.strip().replace("\n", "")
        # Create a pattern dynamically
        pattern_str = r""
        for tag in tags:
            pattern_str += fr"<{tag}>(.*?)</{tag}>\s*"

        # Compile the pattern
        pattern = re.compile(pattern_str, re.DOTALL)

        # Find all matches
        matches = pattern.findall(input_string)

        # If there's only one group of matches (a single match), wrap it in a list
        if isinstance(matches, str):
            matches = [matches]
        
        # If there's only one match and it's a tuple, wrap it to make it a list of tuples
        if matches and isinstance(matches[0], str):
            matches = [matches]

        # Create a list of dictionaries from the matches
        result = [
            {tag: match[i] for i, tag in enumerate(tags)}
            for match in matches
        ]
        return result
    
    def instruct(self, sliced_skills: list[dict]) -> List[dict]:
        parsed_res_list = []
        for i, skill in enumerate(sliced_skills):
            messages = self.construct_prompt_messages(skill)
            res = generate_from_4o_chat_completion(messages=messages, model="gpt-4o-2024-05-13")
            print(res)
            print("====================================")
            parsed_res = self.parse(input_string=res)
            skill["instruction"] = parsed_res[0]["instruction"]
            parsed_res_list.append(skill)
        return parsed_res_list # list here for formatting
    
    def save_as_readable_json(self, res_list: list[dict], path: str):
        readable_res_list = simplify_readable_results(res_list)
        with open(path, "w") as f:
            json.dump(readable_res_list, f, indent=4)
    
    def save(self, res_list: list[dict], path: str):
        with open(path, "wb") as f:
            pickle.dump(res_list, f)

class CodeInstructor():
    def construct_prompt_messages(self, sliced_skill: dict):
        goal = sliced_skill["sub-goal"]
        trajectory = sliced_skill["trajectory"]
        messages = []
        # TODO: format the output better s.t. it can be parsed easily
        system_prompt = f"""\
    You will be given the state-action trajectory of a human user interacting with a web page, a sub-goal that summarizes a part of the trajectory, and a list atomic actions that the original user can take.
    Your task is to:
    Given the sub-goal, summarize the steps to achieve the sub-goal based on the trajectory that can help another agent to achieve the sub-goal. 
    Represent the non-fixed elements (input text, button strings) with descriptive variable names as shown in the example.
    Sub-goal: {goal}

    Atomic actions: 
    11 different types of actions are available.

        noop(wait_ms: float = 1000)
            Description: Do nothing, and optionally wait for the given time (in milliseconds).

        send_msg_to_user(text: str)
            Description: Sends a message to the user.

        scroll(delta_x: float, delta_y: float)
            Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.

        fill(bid: str, value: str)
            Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.

        select_option(bid: str, options: str | list[str])
            Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.

        click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
            Description: Click an element.

        clear(bid: str)
            Description: Clear the input field.

        hover(bid: str)
            Description: Hover over an element.

        go_back()
            Description: Navigate to the previous page in history.

        go_forward()
            Description: Navigate to the next page in history.

        goto(url: str)
            Description: Navigate to a url.

    When outputing, you should follow the format:
    <think>
    think step-by-step, you should firstly locate certain part of trajectory which aims to achieve the sub-goal and then summarize them into a detailed instruction.
    <think>
    <instruction>
    step-by-step detailed instruction to achieve the sub-goal here
    </instruction>
    
    # Example output:
    ## Example 1
    Sub-goal: navigate to the post titled "What is the best book you have ever read?"

    <think>
    This sub-goal 'navigate to the post titled "What is the best book you have ever read?"' is attempted from step 0 to step 2. The user firstly fills out the search bar with the text "What is the best book you have ever read?" and then clicks on the search button. Finally, the user clicks on the most relevant post to navigate to the post.
    </think>
    <instruction>
    1. Fill out the search bar with the text "What is the best book you have ever read?"
    2. Click on the search button
    3. Click on the most relevant post to navigate to the post
    </instruction>
    
    IMPORTANT NOTES you should absolutely follow: 
    1. State is the screenshot of the current webpage. In the screenshot, elements are tagged with their IDs. action is the action taken by the user grounding by the element ID.
    2. DO NOT include any other words except the instruction as the format stated above.
    3. For the instruction, each step should be able to be executed with atomic actions above. Counter examples are like "Identify sth" or "Find sth" because they are not executable with the atomic actions above.
    4. Do not include the exact IDs of the elements in the instruction. You should use the element type or the text of the element to refer to the element.
    5. There could be some noise in the trajectory for the sub-goal. For example, there could be some irrelevant actions that will not contribute to the sub-goal. You should filter out these irrelevant actions when generating the instruction.
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
    
    def parse(self, input_string: str, tags: List[str] = ["instruction"]) -> dict[str, List[str]]:
        # TODO: add error handling

        # strip
        input_string = input_string.strip().replace("\n", "")
        # Create a pattern dynamically
        pattern_str = r""
        for tag in tags:
            pattern_str += fr"<{tag}>(.*?)</{tag}>\s*"

        # Compile the pattern
        pattern = re.compile(pattern_str, re.DOTALL)

        # Find all matches
        matches = pattern.findall(input_string)

        # If there's only one group of matches (a single match), wrap it in a list
        if isinstance(matches, str):
            matches = [matches]
        
        # If there's only one match and it's a tuple, wrap it to make it a list of tuples
        if matches and isinstance(matches[0], str):
            matches = [matches]

        # Create a list of dictionaries from the matches
        result = [
            {tag: match[i] for i, tag in enumerate(tags)}
            for match in matches
        ]
        return result
    
    def instruct(self, sliced_skills: list[dict]) -> List[dict]:
        parsed_res_list = []
        for i, skill in enumerate(sliced_skills):
            messages = self.construct_prompt_messages(skill)
            res = generate_from_4o_chat_completion(messages=messages, model="gpt-4o-2024-05-13")
            print(res)
            print("====================================")
            parsed_res = self.parse(input_string=res)
            skill["instruction"] = parsed_res[0]["instruction"]
            parsed_res_list.append(skill)
        return parsed_res_list # list here for formatting
    
    def save_as_readable_json(self, res_list: list[dict], path: str):
        readable_res_list = simplify_readable_results(res_list)
        with open(path, "w") as f:
            json.dump(readable_res_list, f, indent=4)
    
    def save(self, res_list: list[dict], path: str):
        with open(path, "wb") as f:
            pickle.dump(res_list, f)