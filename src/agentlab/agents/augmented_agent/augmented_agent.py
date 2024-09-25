import traceback
from dataclasses import asdict, dataclass
from warnings import warn
from functools import partial

from browsergym.experiments.loop import AbstractAgentArgs
from langchain.schema import HumanMessage, SystemMessage


from browsergym.experiments.agent import Agent
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.utils import openai_monitored_agent
from agentlab.llm.chat_api import ChatModelArgs
from agentlab.llm.llm_utils import ParseError, RetryError, retry_and_fit, retry
from .augmented_agent_prompt import AugmentedPromptFlags, MainPrompt
import json

from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class AugmentedAgentArgs(AbstractAgentArgs):
    agent_name: str = "AugmentedAgent"
    chat_model_args: ChatModelArgs = None
    flags: AugmentedPromptFlags = None
    max_retry: int = 4

    def make_agent(self, **kwargs):
        return AugmentedAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class AugmentedAgent(Agent):

    def __init__(
        self,
        chat_model_args: ChatModelArgs,
        flags: AugmentedPromptFlags,
        max_retry: int = 4,
    ):

        self.chat_llm = chat_model_args.make_chat_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.action_set = dp.make_action_set(self.flags.action)
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        self.experience = self.load_experience()
        self.embedding_model = INSTRUCTOR('hkunlp/instructor-large')
        self.embedded_experience = self.embed_experience(self.experience)
        self.selected_experience = []
        self._check_flag_constancy()
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @openai_monitored_agent
    def get_action(self, obs):

        self.obs_history.append(obs)

        # retrieve experience if not yet
        if len(self.selected_experience) == 0:
            goal = obs['goal']
            self.selected_experience = self.retrieve_experience(goal)

        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
            experience=self.selected_experience,
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
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long
            if self.flags.use_retry_and_fit:
                ans_dict = retry_and_fit(
                    self.chat_llm,
                    main_prompt=main_prompt,
                    system_prompt=dp.SystemPrompt().prompt,
                    n_retry=self.max_retry,
                    parser=parser,
                    fit_function=fit_function,
                    add_missparsed_messages=self.flags.add_missparsed_messages,
                )
            else:  # classic retry
                prompt = fit_function(shrinkable=main_prompt)

                chat_messages = [
                    SystemMessage(content=dp.SystemPrompt().prompt),
                    HumanMessage(content=prompt),
                ]
                ans_dict = retry(
                    self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser
                )
                # inferring the number of retries, TODO: make this less hacky
                ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
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
    
    def load_experience(self):
        with open('/home/ubuntu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences_json/new_experiences.json', 'r') as f:
            experience = json.load(f)
        return experience
    
    def embed_experience(self, experience):
        embedded_experience = []
        # deduplicate the experience
        experience = list({exp['sub_goal']:exp for exp in experience
                            }.values())
        prefix = "Represent the task goal for retrieval: "
        for exp in experience:
            embedded_experience.append([prefix, exp['sub_goal']])
        return embedded_experience
    
    def retrieve_experience(self, goal, top_k=10):
        query = [['Represent the task goal for retrieving relevant tasks: ', goal]]
        query_embedding = self.embedding_model.encode(query)
        experience_embedding = self.embedding_model.encode(self.embedded_experience)
        similarities = cosine_similarity(query_embedding, experience_embedding)
        # Get the indices of the top-k most similar documents
        top_k_indices = np.argsort(similarities[0])[::-1][:top_k]
        selected_experience = [self.experience[i] for i in top_k_indices]
        return selected_experience

