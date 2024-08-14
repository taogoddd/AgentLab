import os
from openai import AzureOpenAI

client = AzureOpenAI()

# now for azure openai
def generate_from_4o_chat_completion(
    messages: list[dict[str, str]],
    model: str,
) -> str:
    if "AZURE_OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "AZURE_OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    
    response = client.chat.completions.create(  # type: ignore
        model=model,
        messages=messages,
    )
    answer: str = response.choices[0].message.content
    return answer