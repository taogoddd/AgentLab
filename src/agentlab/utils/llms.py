import os
from openai import AzureOpenAI, OpenAI
import openai
from typing import Any
import random
import time

_client: openai.OpenAI | openai.AzureOpenAI = None
def get_openai_client():
    global _client
    if "OPENAI_API_KEY" not in os.environ and "AZURE_OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    if not _client:
        if "AZURE_OPENAI_API_KEY" in os.environ:
            _client = openai.AzureOpenAI()
        elif "OPENAI_API_KEY" in os.environ:
            _client = openai.OpenAI()
    return _client

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
            except Exception as e:
            # except errors as e:
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
            # except Exception as e:
            #     raise e

    return wrapper

@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:

    client = get_openai_client()
    # check whether the client is AzureOpenAI or OpenAI
    if isinstance(client, AzureOpenAI):
        # map the name of the model to the model id
        if model == "gpt-4o":
            model = "gpt-4o-2024-05-13"
        response = client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif isinstance(client, OpenAI):
        response = client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    answer: str = response.choices[0].message.content
    return answer

# now for azure openai
@retry_with_exponential_backoff
def generate_from_4o_chat_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-2024-05-13",
    temperature: float = 1.0,
) -> str:
    
    if "AZURE_OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
    elif "AZURE_OPENAI_API_KEY" in os.environ:
        client = AzureOpenAI()
    
    response = client.chat.completions.create(  # type: ignore
        model=model,
        messages=messages,
        temperature=temperature,
    )
    answer: str = response.choices[0].message.content
    return answer