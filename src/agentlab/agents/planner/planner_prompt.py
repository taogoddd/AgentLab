from dataclasses import dataclass
import logging
from browsergym.core import action
from browsergym.core.action.base import AbstractActionSet
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.dynamic_prompting import PromptElement
from agentlab.llm.llm_utils import parse_html_tags_raise

from agentlab.llm.llm_utils import (
    ParseError,
    count_tokens,
    image_to_jpg_base64_url,
    parse_html_tags_raise,
    extract_code_blocks,
)

@dataclass
class PlannerPromptFlags(dp.Flags):
    """
    A class to represent various flags used to control features in an application.

    Attributes:
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
    """

    obs: dp.ObsFlags
    action: dp.ActionFlags
    use_history: bool = True  
    use_thinking: bool = True
    use_think_history: bool = True
    use_action_history: bool = True
    use_memory: bool = False  #
    memory_window_size: int = 5
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    use_retry_and_fit: bool = False

class SystemPrompt(PromptElement):
    _prompt = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You need to generate a one-step plan that indicates the next step to do to solve the task."""

class MainPrompt(dp.Shrinkable):
    def __init__(
        self,
        history: list[dict],
        flags: PlannerPromptFlags,
        obs: dict,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = History(history, flags.obs, flags.memory_window_size)
        self.instructions = GoalInstructions(
            history[-1]["goal"], extra_instructions=flags.extra_instructions
        )
        self.plan = Plan()
        self.obs = dp.Observation(history[-1], self.flags.obs)
        self.think = Think(visible=lambda: flags.use_thinking)
        self.hints = dp.Hints(visible=lambda: flags.use_hints)
        self.memory = Memory(visible=lambda: flags.use_memory)
        self.examples = Examples(flags=flags)

    @property
    def _prompt(self) -> str:
        # move the hint prompt from after action_prompt to the end
        prompt = f"""\
{self.instructions.prompt}\
{self.obs.prompt}\
{self.history.prompt}\
"""
        
        if self.flags.use_abstract_example:
            prompt += f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.examples._abstract_ex}
"""
        
        if self.flags.use_concrete_example:
            prompt += f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.examples._concrete_ex}
"""
        prompt += self.hints.prompt
        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.think.parse_answer(text_answer))
        ans_dict.update(self.plan.parse_answer(text_answer))
        return ans_dict

class Examples(dp.PromptElement):
    def __init__(self, flags: PlannerPromptFlags) -> None:
        self.flags = flags
    
    def _abstract_ex(self) -> str:
        ex = ""
        if self.flags.use_thinking:
            ex += """<think>
Think step by step.
</think>
"""
            ex += """<next-step-plan>
The plan for the next step.
</next-step-plan>
"""
        return ex
    def _concrete_ex(self) -> str:
        ex = ""
        if self.flags.use_thinking:
            ex += """
<think>
In previous steps, I have navigated to the r/book subreddit, the goal of the whole task is to find the user who posted the latest post in the subreddit. So the next step is to find the latest post in the subreddit.
</think>
"""
            ex += """
<next-step-plan>
Find the latest post in the book subreddit.
</next-step-plan>
"""
        return ex

class Think(PromptElement):
    _prompt = ""

    _abstract_ex = """
<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>
"""
    _concrete_ex = """
<think>
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

# the instructions including the goal and extra instructions
class GoalInstructions(dp.PromptElement):
    def __init__(self, goal: str, visible: bool = True, extra_instructions: str | None = None) -> None:
        super().__init__(visible=visible)
        self._prompt = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next step plan to accomplish your goal. Your plan generated will be given to another executor agent as the objective to accomplish, i.e. generate low-level actions to accomplish the sub-goal.
, make sure to follow the formatting instructions.hhhhh

## Goal of the whole task:
{goal}
"""
        if extra_instructions:
            self._prompt += f"""

## Extra instructions:

{extra_instructions}
"""

# the prompt of a single plan step and its state-action history
class HistoryStepPrompt(dp.Shrinkable):
    def __init__(self, step_number: int, plan_step: dict, flags: PlannerPromptFlags) -> None:
        self.flags = flags
        self.plan_step = plan_step
        self.plan = plan_step["plan"]
        self.history = plan_step["history"]
        self.step_number = step_number

    @property
    def _prompt(self) -> str:
        prompt = ""

        prompt += f"## Step {self.step_number} plan: {self.plan}\n"
        
        for i, step in enumerate(self.history):
            if self.flags.use_think_history:
                prompt += f"\n<think>\n{step["think"]}\n</think>\n"
            
            if self.flags.use_action_history:
                prompt += f"\n<action>\n{step["action"]}\n</action>\n"
        
        return prompt

class History(dp.Shrinkable):
    def __init__(self, history: list[dict], flags: PlannerPromptFlags, window_size: int = 5) -> None:
        """
        Args:
            history: list of dictionaries representing the history of the agent
            flags: flags to control the features of the prompt
            window_size: size of the window to keep in memory (in plans)
        """
        super().__init__(visible=lambda: flags.use_history)

        self.flags = flags
        self.history = history
        self.window_size = window_size
        self.history_steps: list[HistoryStepPrompt] = []

        for i in range(max(len(history)-window_size, 0), len(history)):
            self.history_steps.append(HistoryStepPrompt(i, history[i], flags))
    
    def shrink(self):
        for step in self.history_steps:
            step.shrink()
    
    @property
    def _prompt(self) -> str:
        prompts = [f"# History of last {self.window_size} plan with the task:\n", "Note: The Step i plan here is the i-th plan in the history and the action history under it is the actual actions from the executor\n"]
        for step in self.history_steps:
            prompts.append(step.prompt)
        return "\n".join(prompts) + "\n"
        
        
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
