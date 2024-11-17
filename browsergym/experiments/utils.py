import tiktoken
from transformers import AutoTokenizer

def count_tokens(text, model="gpt-4"):
    """Count the number of tokens in a text."""
    if text == None:
        return 0
    if "Llama" in model:
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
        tokens = tokenizer.tokenize(text)
        num_tokens = len(tokens)
        return num_tokens
    return len(tiktoken.encoding_for_model(model).encode(text))


def count_messages_token(messages, model="gpt-4"):
    """Count the number of tokens in a list of messages.

    Args:
        messages (list): a list of messages, each message can be a string or a
            list of dicts or an object with a content attribute.
        model (str): the model to use for tokenization.

    Returns:
        int: the number of tokens.
    """
    token_count = 0
    for message in messages:
        if hasattr(message, "content"):
            message = message.content

        if isinstance(message, str):
            token_count += count_tokens(message, model)
        # handles messages with image content
        elif isinstance(message, (list, tuple)):
            for part in message:
                if not isinstance(part, dict):
                    raise ValueError(
                        f"The message is expected to be a list of dicts, but got list of {type(message)}"
                    )
                if part["type"] == "text":
                    token_count += count_tokens(part["text"], model)
        else:
            raise ValueError(
                f"The message is expected to be a string or a list of dicts, but got {type(message)}"
            )
    return token_count