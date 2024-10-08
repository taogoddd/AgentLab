"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any

import aiolimiter
import openai
from tqdm.asyncio import tqdm_asyncio

_client: openai.OpenAI | openai.AzureOpenAI = None
def get_openai_client():
    global _client
    if "OPENAI_API_KEY" not in os.environ and "AZURE_OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    if not _client:
        if "OPENAI_API_KEY" in os.environ:
            _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], organization=os.environ.get("OPENAI_ORGANIZATION", ""))
        elif "AZURE_OPENAI_API_KEY" in os.environ:
            _client = openai.AzureOpenAI()
    return _client

_aclient: openai.AsyncOpenAI = None
def get_openai_aclient():
    global _aclient
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    if not _aclient:
        _aclient = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], organization=os.environ.get("OPENAI_ORGANIZATION", ""))
    return _aclient

def get_api_configs_from_env() -> list[dict[str, str]]:
    """Retrieve API configurations (endpoint, key, version) from environment variable and return as a list of dicts."""
    azure_flag = "AZURE_OPENAI_API_CONFIGS" in os.environ
    if "AZURE_OPENAI_API_CONFIGS" not in os.environ and "OPENAI_API_CONFIGS" not in os.environ:
        raise ValueError(
            "AZURE_OPENAI_API_CONFIGS or OPENAI_API_CONFIGS environment variable must be set when using OpenAI API."
        )
    elif "AZURE_OPENAI_API_CONFIGS" in os.environ:
        api_configs_str = os.getenv("AZURE_OPENAI_API_CONFIGS", "")
    elif "OPENAI_API_CONFIGS" in os.environ:
        api_configs_str = os.getenv("OPENAI_API_CONFIGS", "")
    api_configs_list = api_configs_str.split(";")
    
    # Parse each configuration and store in a list of dictionaries
    api_configs = []
    for config in api_configs_list:
        if azure_flag:
            endpoint, api_key, version = config.split(",")
            api_configs.append({
            "endpoint": endpoint.strip(),
            "api_key": api_key.strip(),
            "version": version.strip()
        })
        else:
            api_key = config
            api_configs.append({
                "api_key": api_key.strip()
            })
    
    return api_configs

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (openai.RateLimitError, openai.NotFoundError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

def retry_with_config_rotation(  # type: ignore
    func,
    max_retries: int = 10,
    errors: tuple[Any] = (openai.RateLimitError, openai.NotFoundError),
):
    """Retry a function by rotating API configurations (endpoint, key, version) on failure."""
    
    def wrapper(*args, **kwargs):  # type: ignore
        # Retrieve API configurations
        api_configs = get_api_configs_from_env()

        # Initialize variables
        num_retries = 0

        # Randomly pick a starting index
        config_index = random.randint(0, len(api_configs) - 1)

        # # Get the starting index for the API configuration
        # config_index = os.environ.get("API_CONFIG_START_INDEX", 0)
        # if config_index.isdigit():
        #     config_index = int(config_index)
        # else:
        #     logging.warning(
        #         f"API_CONFIG_START_INDEX is not a valid integer. Using the default value of 0."
        #     )
        #     config_index = 0

        # Loop until a successful response or max_retries is hit
        while True:
            # Set the API key, endpoint, and version from the current config
            current_config = api_configs[config_index]
            if "AZURE_OPENAI_API_CONFIGS" in os.environ:
                os.environ["AZURE_OPENAI_API_KEY"] = current_config["api_key"]
                os.environ["AZURE_OPENAI_ENDPOINT"] = current_config["endpoint"]
                os.environ["OPENAI_API_VERSION"] = current_config["version"]
            elif "OPENAI_API_CONFIGS" in os.environ:
                os.environ["OPENAI_API_KEY"] = current_config["api_key"]
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Increment retries
                num_retries += 1
                print(f"Error occurred: {e}. Retrying with a new API configuration.")
                
                # Rotate to the next configuration
                config_index = (config_index + 1) % len(api_configs)
                
                # Check if max retries have been reached
                if num_retries > max_retries:
                    logging.warning("Maximum number of retries exceeded.")
                    break
                
                # Sleep before retrying
                time.sleep(1)
            # except Exception as e:
            #     # Raise exceptions for any other errors
            #     print(f"Error occurred: {e}.")
            #     time.sleep(1)

    return wrapper

async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    aclient = get_openai_aclient()
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.completions.create(# type: ignore
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return None


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x.choices[0].text for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    client = get_openai_client()
    response = client.completions.create(# type: ignore
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    answer: str = response.choices[0].text
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    aclient = get_openai_aclient()
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.chat.completions.create(# type: ignore
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return None


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x.choices[0].message.content for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    context_length: int | None = None,
    stop_token: str | None = None,
) -> str:
    client = get_openai_client()
    # this is for Azure OpenAI API
    if "gpt-4o" in model and isinstance(client, openai.AzureOpenAI):
        model = "gpt-4o-2024-05-13"

    # Prepare the arguments dynamically
    params = {
        "model": model,
        "messages": messages
    }
    
    # Only add parameters if they are not None
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if top_p is not None:
        params["top_p"] = top_p
    if stop_token is not None:
        params["stop"] = [stop_token]

    response = client.chat.completions.create(**params)  # type: ignore
    answer: str = response.choices[0].message.content
    return answer

@retry_with_config_rotation
def full_generate_from_openai_chat_completion_with_key_pool(
    messages: list[dict[str, str]],
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    context_length: int | None = None,
    stop_token: str | None = None,
    n: int | None = None,
) -> str:
    client = get_openai_client()
    # this is for Azure OpenAI API
    if "gpt-4o" in model and isinstance(client, openai.AzureOpenAI):
        model = "gpt-4o-2024-05-13"

    # Prepare the arguments dynamically
    params = {
        "model": model,
        "messages": messages
    }
    
    # Only add parameters if they are not None
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if top_p is not None:
        params["top_p"] = top_p
    if stop_token is not None:
        params["stop"] = [stop_token]
    if n is not None:
        params["n"] = n

    response = client.chat.completions.create(**params)  # type: ignore
    return response

@retry_with_config_rotation
def generate_from_openai_chat_completion_with_key_pool(
    messages: list[dict[str, str]],
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    context_length: int | None = None,
    stop_token: str | None = None,
) -> str:
    client = get_openai_client()
    # this is for Azure OpenAI API
    if "gpt-4o" in model and isinstance(client, openai.AzureOpenAI):
        model = "gpt-4o-2024-05-13"

    # Prepare the arguments dynamically
    params = {
        "model": model,
        "messages": messages
    }
    
    # Only add parameters if they are not None
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if top_p is not None:
        params["top_p"] = top_p
    if stop_token is not None:
        params["stop"] = [stop_token]

    response = client.chat.completions.create(**params)  # type: ignore
    answer: str = response.choices[0].message.content
    return answer

@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer
