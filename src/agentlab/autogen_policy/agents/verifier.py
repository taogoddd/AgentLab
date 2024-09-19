from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_results, get_website_name_from_url

class Verifier():
    def construct_prompt_messages(self, experience):
        # TODO: verify the success of the sub-goal through the trajectory
        goal = experience["sub-goal"]
        instruction = experience["instruction"]
        website_name = get_website_name_from_url(experience["trajectory"][0]["obs"]["url"])

        system_prompt = f"""\
    You will be given a goal of a task on {website_name} and the corresponding step-by-step instruction to achieve the goal.
    Your task is to verify whether the given pair fits the requirements provided.

    Requirements:
    1. The sub-goal should be a single task and should not contain multiple tasks. For example, "Go to subreddit r/books and upvote the first post" is invalid since it can be split into two sub-goals: "Go to subreddit r/books" and "Upvote the first post".
    2. The instruction should be a step-by-step detailed instruction that can help another agent to achieve the sub-goal. Each step should be a single action that can be executed on the webpage.

    You should follow the format requirements:
    1. if both the sub-goal and instruction are valid, output:
    <result>
    valid
    </result>
    <reasons>
    None
    </reasons>
    2. if sub-goal or instruction is invalid, output:
    <result>
    invalid
    </result>
    <reasons>
    reason for invalidity
    </reasons>

    for example:
    input: 
    sub-goal: 
    Go to subreddit r/books and upvote the first post
    instruction:
    1. Type r/books in the search box
    2. Click search button
    3. Click the first post voting button

    output:
    <result>
    invalid
    </result>
    <reasons>
    The sub-goal contains 2 tasks: "Go to subreddit r/books" and "Upvote the first post"
    </reasons>
    """
        human_prompt = f"""\
    sub-goal:
    {goal}
    instruction:
    {instruction}
    """
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": human_prompt
            }
        ]
        return messages