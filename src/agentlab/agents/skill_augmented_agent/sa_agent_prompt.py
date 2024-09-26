from dataclasses import dataclass
import logging
from browsergym.core import action
from browsergym.core.action.base import AbstractActionSet
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import (
    ParseError,
    count_tokens,
    image_to_jpg_base64_url,
    parse_html_tags_raise,
    extract_code_blocks,
    parse_html_tags_raise
)
from PIL import Image
import requests

@dataclass
class SkillAugmentedPromptFlags(dp.Flags):
    """
    A class to represent various flags used to control features in an application.

    Attributes:
        use_plan (bool): Ask the LLM to provide a plan.
        use_criticise (bool): Ask the LLM to first draft and criticise the action before producing it.
        use_thinking (bool): Enable a chain of thoughts.
        use_concrete_example (bool): Use a concrete example of the answer in the prompt for a generic task.
        use_abstract_example (bool): Use an abstract example of the answer in the prompt.
        use_hints (bool): Add some human-engineered hints to the prompt.
        enable_chat (bool): Enable chat mode, where the agent can interact with the user.
        max_prompt_tokens (int): Maximum number of tokens allowed in the prompt.
        be_cautious (bool): Instruct the agent to be cautious about its actions.
        extra_instructions (Optional[str]): Extra instructions to provide to the agent.
        add_missparsed_messages (bool): When retrying, add the missparsed messages to the prompt.
        use_retry_and_fit (bool): Use the retry_and_fit function that shrinks the prompt at each retry iteration.
        skill_str (str): A string to add to the prompt to indicate the augmented skills of the agent.
    """

    obs: dp.ObsFlags
    action: dp.ActionFlags
    use_plan: bool = False  #
    use_criticise: bool = False  #
    use_thinking: bool = False
    use_reminder: bool = True
    use_memory: bool = False  #
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    use_retry_and_fit: bool = False
    skill_str: str = None

class MainPrompt(dp.Shrinkable):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: SkillAugmentedPromptFlags,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = dp.GoalInstructions(
                obs_history[-1]["goal"], extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(obs_history[-1], self.flags.obs)

        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

        self.reminder = Reminder(obs_history=obs_history, action_history=actions, visible=lambda: flags.use_reminder)

        def time_for_caution():
            # no need for caution if we're in single action mode
            return flags.be_cautious and (
                flags.action.multi_actions or flags.action.action_set == "python"
            )

        self.be_cautious = dp.BeCautious(visible=time_for_caution)
        self.think = Think(visible=lambda: flags.use_thinking)
        self.hints = Hints(visible=lambda: flags.use_hints)
        self.plan = Plan(previous_plan, step, lambda: flags.use_plan)  # TODO add previous plan
        self.criticise = Criticise(visible=lambda: flags.use_criticise)
        self.memory = Memory(visible=lambda: flags.use_memory)

        self.obs_history = obs_history

    @property
    def _prompt(self) -> str:
        # move the hint prompt from after action_prompt to the end
        prefix_prompt = f"""\
{self.instructions.prompt}\
Images together with the goal are shown below. Make sure to refer to them when solving the task.
"""     
        prefix_messages = [{"type": "text", "text": prefix_prompt}]

        img_urls = []
        current_obs = self.obs_history[-1]
        if "goal_image_urls" in current_obs:
                goal_image_urls = current_obs["goal_image_urls"]
                # get the Image object from the goal_image_urls
                for url in goal_image_urls:
                    if url.startswith("http"):
                        input_image = Image.open(requests.get(url, stream=True).raw)
                    else:
                        input_image = Image.open(url)
                
                    img_urls.append(image_to_jpg_base64_url(input_image))
        for i, img_url in enumerate(img_urls):
            prefix_messages.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                    }
                }
            )
        
        prefix_obs_prompt = f"""\
# Observation:
"""
        url = dp.URL(current_obs["url"], True, "## ")
        prefix_obs_prompt += url.prompt
        prefix_obs_prompt += "\n## Screenshot of the page (operatable elements are marked with their bid):"

        
        obs_messages = [{"type": "text", "text": prefix_obs_prompt}]
        obs_messages.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(current_obs["screenshot_som"])
                }
            }
        )
        suffix_obs_prompt = f"""\
## Text observation:
Note: 
-- This lists the bid ids of all interactable elements on the current web page with their text content if any, in the format [bid] [tagType] [text content]. tagType is the type of the element, such as button, link, or textbox. text content is the text content of the element. For example, [1234] [BUTTON] ['Add to Cart'] means that there is a button with id 1234 and text content 'Add to Cart' on the current web page. [] [StaticText] [text] means that the element is of some text that is not interactable.
-- [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.
-- Elements here are all in the screenshot of the current webpage with same bid marked on them.

"""
        som_axtree = current_obs["som_axtree_str"]
        suffix_obs_prompt += som_axtree

        obs_messages.append(
            {
                "type": "text",
                "text": suffix_obs_prompt
            }
        )

        prompt = f"""\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.be_cautious.prompt}\
{self.think.prompt}\
{self.plan.prompt}\
{self.memory.prompt}\
{self.criticise.prompt}\
"""

        if self.flags.use_abstract_example:
            prompt += f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.plan.abstract_ex}\
{self.think.abstract_ex}\
{self.memory.abstract_ex}\
{self.criticise.abstract_ex}\
{self.action_prompt.abstract_ex}\
"""

        if self.flags.use_concrete_example:
            prompt += f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.plan.concrete_ex}\
{self.think.concrete_ex}\
{self.memory.concrete_ex}\
{self.criticise.concrete_ex}\
{self.action_prompt.concrete_ex}\

"""
        prompt += self.reminder.prompt
        prompt += self.hints.prompt

        other_messages = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        full_messages = prefix_messages + obs_messages + other_messages

        # return self.obs.add_screenshot(prompt)
        return full_messages

    def shrink(self):
        self.history.shrink()
        # self.obs.shrink()

    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.think.parse_answer(text_answer))
        ans_dict.update(self.plan.parse_answer(text_answer))
        ans_dict.update(self.memory.parse_answer(text_answer))
        ans_dict.update(self.criticise.parse_answer(text_answer))
        ans_dict.update(self.action_prompt.parse_answer(text_answer))
        return ans_dict


class Memory(dp.PromptElement):
    _prompt = ""  # provided in the abstract and concrete examples

    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions. Some tasks require to
remember hints from previous steps in order to solve it.
</memory>
"""

    _concrete_ex = """
<memory>
I clicked on bid "32" to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
</memory>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["memory"], merge_multiple=True)


class Plan(dp.PromptElement):
    def __init__(self, previous_plan, plan_step, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self.previous_plan = previous_plan
        self._prompt = f"""
# Plan:

You just executed step {plan_step} of the previously proposed plan:\n{previous_plan}\n
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary.
"""

    _abstract_ex = """
<plan>
Provide a multi step plan that will guide you to accomplish the goal. There
should always be steps to verify if the previous action had an effect. The plan
can be revisited at each steps. Specifically, if there was something unexpected.
The plan should be cautious and favor exploring befor submitting.
</plan>

<step>Integer specifying the step of current action
</step>
"""

    _concrete_ex = """
<plan>
1. fill form (failed)
    * type first name
    * type last name
2. Try to activate the form
    * click on tab 2
3. fill form again
    * type first name
    * type last name
4. verify and submit
    * verify form is filled
    * submit if filled, if not, replan
</plan>

<step>2</step>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["plan", "step"])


class Criticise(dp.PromptElement):
    _prompt = ""

    _abstract_ex = """
<action_draft>
Write a first version of what you think is the right action.
</action_draft>

<criticise>
Criticise action_draft. What could be wrong with it? Enumerate reasons why it
could fail. Did your past actions had the expected effect? Make sure you're not
repeating the same mistakes.
</criticise>
"""

    _concrete_ex = """
<action_draft>
click("32")
</action_draft>

<criticise>
click("32") might not work because the element is not visible yet. I need to
explore the page to find a way to activate the form.
</criticise>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["action_draft", "criticise"])


if __name__ == "__main__":
    html_template = """
    <html>
    <body>
    <div>
    Hello World.
    Step {}.
    </div>
    </body>
    </html>
    """

    OBS_HISTORY = [
        {
            "goal": "do this and that",
            "pruned_html": html_template.format(1),
            "axtree_txt": "[1] Click me",
            "last_action_error": "",
            "focused_element_bid": "32",
        },
        {
            "goal": "do this and that",
            "pruned_html": html_template.format(2),
            "axtree_txt": "[1] Click me",
            "last_action_error": "",
            "focused_element_bid": "32",
        },
        {
            "goal": "do this and that",
            "pruned_html": html_template.format(3),
            "axtree_txt": "[1] Click me",
            "last_action_error": "Hey, there is an error now",
            "focused_element_bid": "32",
        },
    ]
    ACTIONS = ["click('41')", "click('42')"]
    MEMORIES = ["memory A", "memory B"]
    THOUGHTS = ["thought A", "thought B"]

    flags = dp.ObsFlags(
        use_html=True,
        use_ax_tree=True,
        use_plan=True,
        use_criticise=True,
        use_thinking=True,
        use_error_logs=True,
        use_past_error_logs=True,
        use_history=True,
        use_action_history=True,
        use_memory=True,
        use_diff=True,
        html_type="pruned_html",
        use_concrete_example=True,
        use_abstract_example=True,
        multi_actions=True,
        use_screenshot=False,
    )

    print(
        MainPrompt(
            action_set=dp.make_action_set(
                "bid", is_strict=False, multiaction=True, demo_mode="off"
            ),
            obs_history=OBS_HISTORY,
            actions=ACTIONS,
            memories=MEMORIES,
            thoughts=THOUGHTS,
            previous_plan="No plan yet",
            step=0,
            flags=flags,
        ).prompt
    )

class Think(dp.PromptElement):
    _prompt = ""

    _abstract_ex = """
<think>
Think step by step. Firstly check whether the shortcuts and skills given can help. If so, refer to it. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>
"""
    _concrete_ex = """
<think>
After reviewing shortcuts and skills given, to set the year, I need to click the drop down and then select the year.
From previous action I tried to set the value of year to "2022",
using select_option, but it doesn't appear to be in the form. It may be a
dynamic dropdown, I will try using click with the bid "a324" and look at the
response from the page.
</think>
"""
    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["think"], merge_multiple=True)
        except ParseError as e:
            return {"think": text_answer, "parse_error": str(e)}

class ActionPrompt(dp.PromptElement):

    _concrete_ex = """
<action>
click('a324')
</action>
"""

    def __init__(self, action_set: AbstractActionSet, action_flags: dp.ActionFlags) -> None:
        super().__init__()
        self.action_set = action_set
        self.action_flags = action_flags
        action_set_generic_info = """\
Note: This action set allows you to interact with your environment. Most of them
are python function executing playwright code. The primary way of referring to
elements in the page is through bid which are specified in your observations.

"""
        action_description = action_set.describe(
            with_long_description=action_flags.long_description,
            with_examples=action_flags.individual_examples,
        )
        self._prompt = (
            f"# Action space:\n{action_set_generic_info}{action_description}{dp.MacNote().prompt}\n"
        )
        self._abstract_ex = f"""
<action>
{self.action_set.example_action(abstract=True)}
</action>
"""

    #         self._concrete_ex = f"""
    # <action>
    # {self.action_set.example_action(abstract=False)}
    # </action>
    # """

    def _parse_answer(self, text_answer):
        try:
            ans_dict = parse_html_tags_raise(text_answer, keys=["action"], merge_multiple=True)
        except ParseError as e:
            if self.action_flags.is_strict:
                raise e
            else:
                # try to extract code blocks
                blocks = extract_code_blocks(text_answer)
                if len(blocks) == 0:
                    raise e
                else:
                    code = "\n".join([block for _, block in blocks])
                    ans_dict = {"action": code, "parse_error": str(e)}

        try:
            # just check if action can be mapped to python code but keep action as is
            # the environment will be responsible for mapping it to python
            self.action_set.to_python_code(ans_dict["action"])
        except Exception as e:
            raise ParseError(
                f"Error while parsing action\n: {e}\n"
                "Make sure your answer is restricted to the allowed actions."
            )

        return ans_dict

class Hints(dp.PromptElement):
    """Not super useful and stale."""

    # NOTE: are these hints still relevant?

    # MOD: Delete unrelevant hints for WebArena tasks

#     _prompt = """\
# Notes you MUST follow:
# * Refer the provided shortcuts and skills to complete the task if they can help for the task.
# * Make sure to use bid to identify elements when using commands.
# * If you need to select an option, you may use select_option() to do this if you know the options values. Otherwise, click on the dropdown to view the options and then use select_option() to select the option.
# """
    # remove highlight skills
    _prompt = """\
Note:
* Make sure to use bid to identify elements when using commands.
* If you need to select an option, you may use select_option() to do this if you know the options values. Otherwise, click on the dropdown to view the options and then use select_option() to select the option.
"""

class Reminder(dp.PromptElement):
    max_repetitive_actions = 3
    
    def __init__(self, obs_history: list[dict], action_history: list[str], visible) -> None:
        super().__init__(visible=visible)
        self.obs_history = obs_history
        self.action_history = action_history
        # check if the last n actions were the same with each other
        self.repetitive_action_flag = len(set(action_history[-self.max_repetitive_actions:])) == 1 if len(action_history) >= self.max_repetitive_actions else False
        # check if the last action had no effect on the page
        if len(obs_history) >= 2 and "axtree_txt" in obs_history[-1] and "axtree_txt" in obs_history[-2]:
            self.no_effect_action_flag = obs_history[-1]["axtree_txt"] == obs_history[-2]["axtree_txt"]
        else:
            self.no_effect_action_flag = False
        self.repetitive_action_reminder = f"""\
## Repetitive Actions Reminder
The last {self.max_repetitive_actions} actions were repetitive. All of them are {action_history[-1]}. You probably get stuck in a loop. Make sure to review the observation, goal and history and think about the next action.
""" if self.repetitive_action_flag else ""
        self.no_effect_action_reminder = f"""\
## No Effect Action Reminder
The last action you took had no effect on the page, i.e. the page does not change. If this is a scroll action, this means it is already the buttom of the page, if you can not still find results you want, you may need to modify the search query or go to somewhere else to find your answer. For other actions, this might be because the element is not operatable or the option you select does not exist. Make sure to review the observation, goal and history and think about the next action.
""" if self.no_effect_action_flag else ""
        self._prompt = f"""\
# Reminder
{"No reminder" if not self.repetitive_action_flag and not self.no_effect_action_flag else ""}
{self.repetitive_action_reminder}
{self.no_effect_action_reminder}
"""