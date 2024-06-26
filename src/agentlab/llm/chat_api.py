from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re

from langchain.schema import AIMessage
import logging

from langchain_openai import ChatOpenAI
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class CheatMiniWoBLLM:
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    def invoke(self, messages) -> str:
        prompt = messages[-1].content
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        answer = f"""I'm clicking the button as requested.
<action>
{action}
</action>
"""
        return AIMessage(content=answer)

    def __call__(self, messages) -> str:
        return self.invoke(messages)


@dataclass
class CheatMiniWoBLLMArgs:
    model_name = "cheat_miniwob_click_test"
    max_total_tokens = 1024
    max_input_tokens = 1024 - 128
    max_new_tokens = 128
    max_trunk_itr = 10

    def make_chat_model(self):
        return CheatMiniWoBLLM()


@dataclass
class ChatModelArgs(ABC):
    model_name: str
    max_total_tokens: int
    max_input_tokens: int
    max_new_tokens: int
    max_trunk_itr: int = None
    temperature: float = 0.1
    model_url: str = None

    @abstractmethod
    def make_chat_model(self):
        pass

    @abstractmethod
    def prepare_server(self, registry):
        pass

    @abstractmethod
    def close_server(self):
        pass

    def cleanup(self):
        if self.model_url:
            self.close_server()
            self.model_url = None

    def key(self):
        return json.dumps(
            {
                "model_name": self.model_name,
                "max_total_tokens": self.max_total_tokens,
                "max_input_tokens": self.max_input_tokens,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        )


@dataclass
class OpenAIChatModelArgs(ChatModelArgs):
    vision_support: bool = False

    def make_chat_model(self):
        model_name = self.model_name.split("/")[-1]
        return ChatOpenAI(
            model_name=model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

    def prepare_server(self, registry):
        pass

    def close_server(self):
        pass
