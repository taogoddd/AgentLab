import traceback
from dataclasses import asdict, dataclass
from warnings import warn
from functools import partial
from PIL import Image
from io import BytesIO
import base64
from browsergym.experiments.loop import AbstractAgentArgs
from langchain.schema import HumanMessage, SystemMessage


from browsergym.experiments.agent import Agent
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.utils import openai_monitored_agent
from agentlab.llm.chat_api import ChatModelArgs
from agentlab.llm.llm_utils import ParseError, RetryError, retry_and_fit, retry
from .prompt import GenericPromptFlags, MainPrompt

@dataclass
class HumanAnnotatorArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    agent_name: str = "HumanAnnotator"
    flags: GenericPromptFlags = None

    def make_agent(self, **kwargs):
        exp_dir = kwargs.get("exp_dir", None)
        return HumanAnnotator(
            chat_model_args=self.chat_model_args, flags=self.flags, exp_dir=exp_dir
        )

class HumanAnnotator(Agent):

    def __init__(
        self,
        chat_model_args: ChatModelArgs,
        flags: GenericPromptFlags,
        exp_dir: str = None,
    ):

        self.chat_model_args = chat_model_args

        self.flags = flags
        self.action_set = dp.make_action_set(self.flags.action)
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        self.exp_dir = exp_dir

        self._check_flag_constancy()
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def save_annotation_info(self, chat_messages: list):
        '''
        return the readable string of the chat messages
        '''
        text_info_path = self.exp_dir / "annotation_text_info.txt"
        image_info_path = self.exp_dir / "annotation_image_info.jpg"

        chat_message_str = ""
        for message in chat_messages:
            if message.type == "system":
                chat_message_str += "[SYSTEM]\n" + message.content + "\n"
            elif message.type == "human":
                # check whether message.content is a list of messages
                content = message.content
                if isinstance(content, list):
                    content = message.content[0].get("text", "") # only take the text part
                    base64_url = message.content[1]["image_url"]["url"]
                chat_message_str += "-----------------------------------\n"
                chat_message_str += "[USER]\n" + content + "\n"
        
        # save the chat message to the file (overwrite)
        with open(text_info_path, "w") as f:
            f.write(chat_message_str)
        
        # save the base64 string to an jpg image
        
        # remove the base64 header
        base64_string = base64_url.split(",")[-1]

        # Decode the base64 string
        image_data = base64.b64decode(base64_string)

        # Create an image from the decoded bytes
        image = Image.open(BytesIO(image_data))

        # Save the image to a file
        image.save(image_info_path)

        return text_info_path, image_info_path

    @openai_monitored_agent
    def get_action(self, obs):

        self.obs_history.append(obs)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunk_itr = self._get_maxes()

        fit_function = partial(
            dp.fit_tokens,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunk_itr,
        )

        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)
            return ans_dict, True, ""

        try:
            prompt = fit_function(shrinkable=main_prompt)

            chat_messages = [
                SystemMessage(content=dp.SystemPrompt().prompt),
                HumanMessage(content=prompt),
            ]
            human_prompt = "*"*50 + "\n"
            human_prompt += f"Text info path: {self.save_annotation_info(chat_messages)[0]}\nImage info path: {self.save_annotation_info(chat_messages)[1]}\nEnter your action based on the given info: "
            ans_dict = {"action": input(human_prompt)}
        except RetryError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"action": None}

            # TODO Debatable, it shouldn't be reported as some error, since we don't
            # want to re-launch those failure.

            # ans_dict["err_msg"] = str(e)
            # ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry + 1
        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        ans_dict["chat_messages"] = chat_messages
        ans_dict["chat_message_contents"] = [m.content for m in chat_messages]
        return ans_dict["action"], ans_dict

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunk_itr = (
            self.chat_model_args.max_trunk_itr
            if self.chat_model_args.max_trunk_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunk_itr
