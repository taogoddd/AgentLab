from browsergym.experiments.agent import Agent
from agentlab.llm.chat_api import ChatModelArgs
from .planner_prompt import PlannerPromptFlags, MainPrompt, SystemPrompt
from functools import partial
from agentlab.llm.llm_utils import ParseError, RetryError, retry_and_fit, retry
from agentlab.agents import dynamic_prompting as dp
from langchain.schema import HumanMessage, SystemMessage

class Planner(Agent):
    history = [] # record obs, action and plan for each step
    def __init__(self, chat_model_args: ChatModelArgs = None, flags: PlannerPromptFlags = None, max_retry: int = 4):
        self.chat_llm = chat_model_args.make_chat_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry
        self.flags = flags
    
    def update_history(self, trajecotry: dict):
        self.history.append(trajecotry)
    
    def get_plan(self, obs):

        # init step info
        step = {"plan": None, "history": []}

        # the input prompt is organized here
        main_prompt = MainPrompt(
            history=self.history,
            flags=self.flags,
        )
        system_prompt = SystemPrompt().prompt
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
                    system_prompt=system_prompt,
                    n_retry=self.max_retry,
                    parser=parser,
                    fit_function=fit_function,
                    add_missparsed_messages=self.flags.add_missparsed_messages,
                )
            else:  # classic retry
                fitted_main_prompt = fit_function(shrinkable=main_prompt)
                chat_messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=fitted_main_prompt),
                ]
                ans_dict = retry(
                    self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser
                )
                # inferring the number of retries, TODO: make this less hacky
                ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except RetryError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"plan": None}

            # TODO Debatable, it shouldn't be reported as some error, since we don't
            # want to re-launch those failure.

            # ans_dict["err_msg"] = str(e)
            # ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry + 1
        
        # update history
        step["plan"] = ans_dict.get("plan", None)
        self.history.append(step)

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