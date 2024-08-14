from agentlab.utils.llms import generate_from_4o_chat_completion
import base64

def img_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
    return img_base64

def construct_prompt_messages(task_id: int, obs_path: str, url: str, objective: str) -> list:
    messages = []
    system_prompt = """\
You are an autonomous intelligent agent tasked with navigating a web browser to generate a good plan to complete a specific task. You need to generate a high-level action to take in the next step given the information you have to complete the given task.

Here's the information you'll have:
The user's objective: This is the task you're trying to generate plan for.
The observation for the current webpage: This is the screenshot of the current webpage.
The current web page's URL: This is the page you're currently navigating.

To be successful, it is very important to follow the following rules:
1. You should only output one high-level subgoal to achieve in the next few steps at a time.
2. You should follow the examples to format.
"""
    examples = [
        (
            "src/agentlab/planner/examples/example1.png",
            """OBSERVATION:
the screenshot provided for you
URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the size of HP Inkjet Fax Machine
""",
            "[NEXT SUBGOAL]: Navigate to the product details page of HP Inkjet Fax Machine",
        )
    ]
    prompt = f"""OBSERVATION:
the screenshot provided for you
URL: {url}
OBJECTIVE: {objective}
"""
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    for img_path, prompt, action in examples:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_to_base64(img_path)}"
                    }
                },
                {
                    "type": "text",
                    "content": prompt
                }
            ]
        })
        messages.append({
            "role": "assistant",
            "content": action
        })
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_to_base64(obs_path)}"
                }
            },
            {
                "type": "text",
                "content": prompt
            }
        ]
    })
    return messages

task_id = 0
objective = "What is the top-1 best-selling product in 2022"
obs_path = "/home/ytliu/agentlab_results/agentlab_baseline/2024-06-27_11-41-13_GenericAgent_on_webarena.0_51_14d4f1/screenshot_step_0.jpg"
url = "http://localhost:7780/admin/admin/dashboard/"
messages = construct_prompt_messages(task_id, obs_path, url, objective)
model = "gpt-4o-2024-05-13"
response = generate_from_4o_chat_completion(messages, model)
print(response)