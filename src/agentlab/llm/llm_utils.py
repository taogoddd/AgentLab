import collections
import json
import os
from pathlib import Path
import re
import time
from warnings import warn
import logging

from functools import cache
import numpy as np
import tiktoken
import yaml
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from langchain.schema import SystemMessage, HumanMessage
from openai import BadRequestError
from joblib import Memory
from transformers import AutoModel
from transformers import AutoTokenizer
import io
import base64
from PIL import Image
from openai import RateLimitError
from webarena.llms.providers.openai_utils import get_api_configs_from_env


def _extract_wait_time(error_message, min_retry_wait_time=60):
    """Extract the wait time from an OpenAI RateLimitError message."""
    match = re.search(r"try again in (\d+(\.\d+)?)s", error_message)
    if match:
        return max(min_retry_wait_time, float(match.group(1)))
    return min_retry_wait_time


class RetryError(ValueError):
    pass


def retry(
    chat: ChatOpenAI | AzureChatOpenAI,
    messages,
    n_retry,
    parser,
    log=True,
    min_retry_wait_time=60,
    rate_limit_max_wait_time=60 * 30,
):
    """Retry querying the chat models with the response from the parser until it
    returns a valid value.

    If the answer is not valid, it will retry and append to the chat the  retry
    message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be slow
    and expensive.

    Parameters:
    -----------
        chat (function) : a langchain ChatOpenAI taking a list of messages and
            returning a list of answers.
        messages (list) : the list of messages so far.
        n_retry (int) : the maximum number of sequential retries.
        parser (function): a function taking a message and returning a tuple
        with the following fields:
            value : the parsed value,
            valid : a boolean indicating if the value is valid,
            retry_message : a message to send to the chat if the value is not valid
        log (bool): whether to log the retry messages.
        min_retry_wait_time (float): the minimum wait time in seconds
            after RateLimtError. will try to parse the wait time from the error
            message.

    Returns:
    --------
        value: the parsed value
    """
    tries = 0
    rate_limit_total_delay = 0
    api_configs = get_api_configs_from_env()
    config_index = os.environ.get("API_CONFIG_START_INDEX", 0)
    if config_index.isdigit():
        config_index = int(config_index)
    else:
        logging.warning(
            f"API_CONFIG_START_INDEX is not a valid integer. Using the default value of 0."
        )
        config_index = 0

    while tries < n_retry and rate_limit_total_delay < rate_limit_max_wait_time:
        try:
            current_config = api_configs[config_index]
            if "AZURE_OPENAI_API_CONFIGS" in os.environ:
                os.environ["AZURE_OPENAI_API_KEY"] = current_config["api_key"]
                os.environ["AZURE_OPENAI_ENDPOINT"] = current_config["endpoint"]
                os.environ["OPENAI_API_VERSION"] = current_config["version"]
            elif "OPENAI_API_CONFIGS" in os.environ:
                os.environ["OPENAI_API_KEY"] = current_config["api_key"]
            
            # modify the chat object to use the new configuration
            chat.openai_api_key = current_config["api_key"]
            if isinstance(chat, AzureChatOpenAI):
                chat.azure_endpoint = current_config["endpoint"]
                chat.openai_api_version = current_config["version"]
            
            answer = chat.invoke(messages)
        except RateLimitError as e:
            # Rotate to the next configuration
            config_index = (config_index + 1) % len(api_configs)

            wait_time = _extract_wait_time(e.args[0], min_retry_wait_time)
            logging.warning(f"RateLimitError, waiting {wait_time}s before retrying.")
            time.sleep(wait_time)
            rate_limit_total_delay += wait_time
            if rate_limit_total_delay >= rate_limit_max_wait_time:
                logging.warning(
                    f"Total wait time for rate limit exceeded. Waited {rate_limit_total_delay}s > {rate_limit_max_wait_time}s."
                )
                raise
            continue

        messages.append(answer)

        value, valid, retry_message = parser(answer.content)
        if valid:
            return value

        tries += 1
        if log:
            msg = f"Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer.content}\n[User]:\n{retry_message}"
            logging.info(msg)
        messages.append(HumanMessage(content=retry_message))

    raise RetryError(f"Could not parse a valid value after {n_retry} retries.")


def retry_and_fit(
    chat: ChatOpenAI | AzureChatOpenAI,
    main_prompt,
    system_prompt: str,
    n_retry,
    parser,
    log=True,
    min_retry_wait_time=60,
    rate_limit_max_wait_time=60 * 30,
    fit_function: callable = lambda shrinkable, *kw: shrinkable,
    add_missparsed_messages=True,
    examples = []
):
    """Retry querying the chat models with the response from the parser until it
    returns a valid value. The prompt is passed through a fitting function at each
    retry.

    If the answer is not valid, it will retry and append to the chat (depending on
    add_missparsed_messages) the  retry message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be slow
    and expensive.

    Parameters:
    -----------
        chat (function) : a langchain ChatOpenAI taking a list of messages and
            returning a list of answers.
        messages (list) : the list of messages so far.
        n_retry (int) : the maximum number of sequential retries.
        parser (function): a function taking a message and returning a tuple
        with the following fields:
            value : the parsed value,
            valid : a boolean indicating if the value is valid,
            retry_message : a message to send to the chat if the value is not valid
        log (bool): whether to log the retry messages.
        min_retry_wait_time (float): the minimum wait time in seconds
            after RateLimtError. will try to parse the wait time from the error
            message.
        rate_limit_max_wait_time (float): the maximum total wait time in seconds
            for rate limit errors.
        fit_function (callable): a function to fit the tokens before retrying.
            takes main_prompt (str) and additional_prompts (List[str]) and returns
            a new prompt.
        add_missparsed_messages (bool): whether to add the retry message to the
            chat.

    Returns:
    --------
        value: the parsed value
    """
    tries = 0
    rate_limit_total_delay = 0

    api_configs = get_api_configs_from_env()
    config_index = os.environ.get("API_CONFIG_START_INDEX", 0)
    if config_index.isdigit():
        config_index = int(config_index)
    else:
        logging.warning(
            f"API_CONFIG_START_INDEX is not a valid integer. Using the default value of 0."
        )
        config_index = 0

    additional_prompts = []

    while tries < n_retry and rate_limit_total_delay < rate_limit_max_wait_time:
        current_config = api_configs[config_index]
        if "AZURE_OPENAI_API_CONFIGS" in os.environ:
            os.environ["AZURE_OPENAI_API_KEY"] = current_config["api_key"]
            os.environ["AZURE_OPENAI_ENDPOINT"] = current_config["endpoint"]
            os.environ["OPENAI_API_VERSION"] = current_config["version"]
        elif "OPENAI_API_CONFIGS" in os.environ:
            os.environ["OPENAI_API_KEY"] = current_config["api_key"]
        # fit tokens
        prompt = fit_function(
            shrinkable=main_prompt, additional_prompts=[system_prompt] + additional_prompts
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        messages += [HumanMessage(content=content) for content in additional_prompts]

        # modify the chat object to use the new configuration
        chat.openai_api_key = current_config["api_key"]
        if isinstance(chat, AzureChatOpenAI):
            chat.azure_endpoint = current_config["endpoint"]
            chat.openai_api_version = current_config["version"]

        try:
            answer = chat.invoke(messages)
        except RateLimitError as e:
            # Rotate to the next configuration
            config_index = (config_index + 1) % len(api_configs)

            wait_time = _extract_wait_time(e.args[0], min_retry_wait_time)
            logging.warning(f"RateLimitError, waiting {wait_time}s before retrying.")
            time.sleep(wait_time)
            rate_limit_total_delay += wait_time
            if rate_limit_total_delay >= rate_limit_max_wait_time:
                logging.warning(
                    f"Total wait time for rate limit exceeded. Waited {rate_limit_total_delay}s > {rate_limit_max_wait_time}s."
                )
                raise
            continue

        value, valid, retry_message = parser(answer.content)
        if valid:
            value["n_retry"] = tries
            value["chat_messages"] = [m.content for m in messages]
            return value

        tries += 1
        if log:
            msg = f"Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer.content}\n[User]:\n{retry_message}"
            logging.info(msg)
        if add_missparsed_messages:
            additional_prompts.append(answer.content)
            additional_prompts.append(retry_message)

    raise RetryError(f"Could not parse a valid value after {n_retry} retries.")


def retry_parallel(chat: ChatOpenAI | AzureChatOpenAI, messages, n_retry, parser):
    """Retry querying the chat models with the response from the parser until it returns a valid value.

    It will stop after `n_retry`. It assuemes that chat will generate n_parallel answers for each message.
    The best answer is selected according to the score returned by the parser. If no answer is valid, the
    it will retry with the best answer so far and append to the chat the retry message. If there is a
    single parallel generation, it behaves like retry.

    This function is, in principle, more robust than retry. The speed and cost overhead is minimal with
    the prompt is large and the length of the generated message is small.

    Parameters:
    -----------
        chat (function) : a langchain ChatOpenAI taking a list of messages and returning a list of answers.
            The number of parallel generations is specified at the creation of the chat object.
        messages (list) : the list of messages so far.
        n_retry (int) : the maximum number of sequential retries.
        parser (function): a function taking a message and returning a tuple with the following fields:
            value : the parsed value,
            valid : a boolean indicating if the value is valid,
            retry_message : a message to send to the chat if the value is not valid,
            score : a score to select the best answer from the parallel generations

    Returns:
    --------
        value: the parsed value
    """

    for i in range(n_retry):
        try:
            answers = chat.generate([messages]).generations[0]  # chat.n parallel completions
        except BadRequestError as e:
            # most likely, the added messages triggered a message too long error
            # we thus retry without the last two messages
            if i == 0:
                raise e
            msg = f"BadRequestError, most likely the message is too long retrying with previous query."
            warn(msg)
            messages = messages[:-2]
            answers = chat.generate([messages]).generations[0]

        values, valids, retry_messages, scores = zip(
            *[parser(answer.message.content) for answer in answers]
        )
        idx = np.argmax(scores)
        value = values[idx]
        valid = valids[idx]
        retry_message = retry_messages[idx]
        answer = answers[idx].message

        if valid:
            return value

        msg = f"Query failed. Retrying {i+1}/{n_retry}.\n[LLM]:\n{answer.content}\n[User]:\n{retry_message}"
        warn(msg)
        messages.append(answer)  # already of type AIMessage
        messages.append(SystemMessage(content=retry_message))

    raise ValueError(f"Could not parse a valid value after {n_retry} retries.")


def truncate_tokens(text, max_tokens=8000, start=0, model_name="gpt-4"):
    """Use tiktoken to truncate a text to a maximum number of tokens."""
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) - start > max_tokens:
        return enc.decode(tokens[start : (start + max_tokens)])
    else:
        return text


@cache
def get_tokenizer(model_name="openai/gpt-4"):
    logging.debug(f"Loading tokenizer for model {model_name}")
    if model_name == "cheat_miniwob_click_test":
        return tiktoken.encoding_for_model("gpt-4")
    if model_name.startswith("openai") or model_name.startswith("azureopenai"):
        return tiktoken.encoding_for_model(model_name.split("/")[-1])
    if model_name.startswith("reka"):
        logging.warning(
            "Reka models don't have a tokenizer implemented yet. Using the default one."
        )
        return tiktoken.encoding_for_model("gpt-4")
    else:
        return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text, model="openai/gpt-4"):
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def json_parser(message):
    """Parse a json message for the retry function."""

    try:
        value = json.loads(message)
        valid = True
        retry_message = ""
    except json.JSONDecodeError as e:
        warn(e)
        value = {}
        valid = False
        retry_message = "Your response is not a valid json. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def yaml_parser(message):
    """Parse a yaml message for the retry function."""

    # saves gpt-3.5 from some yaml parsing errors
    message = re.sub(r":\s*\n(?=\S|\n)", ": ", message)

    try:
        value = yaml.safe_load(message)
        valid = True
        retry_message = ""
    except yaml.YAMLError as e:
        warn(str(e))
        value = {}
        valid = False
        retry_message = "Your response is not a valid yaml. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def _compress_chunks(text, identifier, skip_list, split_regex="\n\n+"):
    """Compress a string by replacing redundant chunks by identifiers. Chunks are defined by the split_regex."""
    text_list = re.split(split_regex, text)
    text_list = [chunk.strip() for chunk in text_list]
    counter = collections.Counter(text_list)
    def_dict = {}
    id = 0

    # Store items that occur more than once in a dictionary
    for item, count in counter.items():
        if count > 1 and item not in skip_list and len(item) > 10:
            def_dict[f"{identifier}-{id}"] = item
            id += 1

    # Replace redundant items with their identifiers in the text
    compressed_text = "\n".join(text_list)
    for key, value in def_dict.items():
        compressed_text = compressed_text.replace(value, key)

    return def_dict, compressed_text


def compress_string(text):
    """Compress a string by replacing redundant paragraphs and lines with identifiers."""

    # Perform paragraph-level compression
    def_dict, compressed_text = _compress_chunks(
        text, identifier="§", skip_list=[], split_regex="\n\n+"
    )

    # Perform line-level compression, skipping any paragraph identifiers
    line_dict, compressed_text = _compress_chunks(
        compressed_text, "¶", list(def_dict.keys()), split_regex="\n+"
    )
    def_dict.update(line_dict)

    # Create a definitions section
    def_lines = ["<definitions>"]
    for key, value in def_dict.items():
        def_lines.append(f"{key}:\n{value}")
    def_lines.append("</definitions>")
    definitions = "\n".join(def_lines)

    return definitions + "\n" + compressed_text


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


class ParseError(Exception):
    pass


def extract_code_blocks(text) -> list[tuple[str, str]]:
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

    matches = pattern.findall(text)
    return [(match[0], match[1].strip()) for match in matches]


def parse_html_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_html_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_html_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.
    optional_keys : list of str
        The HTML tags to extract the content from, but are optional.

    Returns
    -------
    dict
        A dictionary mapping each key to subset of `text` that match the key.
    bool
        Whether the parsing was successful.
    str
        A message to be displayed to the agent if the parsing was not successful.
    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if not key in content_dict:
            if not key in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


class ChatCached:
    # I wish I could extend ChatOpenAI, but it is somehow locked, I don't know if it's pydantic soercey.

    def __init__(self, chat, memory=None):
        self.chat = chat
        self.memory = memory if memory else Memory(location=Path.home() / "llm-cache", verbose=10)
        self._call = self.memory.cache(self.chat.__call__, ignore=["self"])
        self._generate = self.memory.cache(self.chat.generate, ignore=["self"])

    def __call__(self, messages):
        return self._call(messages)

    def generate(self, messages):
        return self._generate(messages)


def download_and_save_model(model_name: str, save_dir: str = "."):
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model downloaded and saved to {save_dir}")


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"


if __name__ == "__main__":

    # model_to_download = "THUDM/agentlm-70b"
    model_to_download = "databricks/dbrx-instruct"
    save_dir = "/mnt/ui_copilot/data_rw/base_models/"
    # set the following env variable to enable the transfer of the model
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    download_and_save_model(model_to_download, save_dir=save_dir)
