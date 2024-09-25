from agentlab.autogen_policy.agents import Summarizer, Abstractor, Verifier, Refiner, Locator, Slicer, Instructor, RLocator
from agentlab.autogen_policy.utils.utils import Obs, ProcessedObs, TrajectoryStep, img_array_to_base64, simplify_readable_trajectory, get_trajectory_from_annotation, get_website_name_from_url, generate_from_4o_chat_completion

import json
import pickle
import os
ROOT_PATH = "/home/ubuntu/github/AgentLab/src/agentlab/autogen_policy"

def prepare_skill_dir(skill_type: str):
    # check if the directory exists
    if not os.path.exists(f"{ROOT_PATH}/skills/{skill_type}/json"):
        os.makedirs(f"{ROOT_PATH}/skills/{skill_type}/json")
    
    if not os.path.exists(f"{ROOT_PATH}/skills/{skill_type}/pkl"):
        os.makedirs(f"{ROOT_PATH}/skills/{skill_type}/pkl")


def skill_learning(traj_path: str):
    # extract task_name from traj_path
    task_name = traj_path.split("/")[-1]

    trajectory = get_trajectory_from_annotation(traj_path)

    # slice the trajectory
    slicer = Slicer()
    sliced_skills = slicer.slice(trajectory)
    slicer.save_as_readable_json(sliced_skills, f"{ROOT_PATH}/skills/sliced_skills/json/{task_name}.json")
    slicer.save(sliced_skills, f"{ROOT_PATH}/skills/sliced_skills/pkl/{task_name}.pkl")

    # generate instruction for sliced skills
    skill_type = "instructed_skills"
    sliced_skills = pickle.load(open(f"{ROOT_PATH}/skills/sliced_skills/pkl/{task_name}.pkl", "rb"))
    instructor = Instructor()
    instructed_skills = instructor.instruct(sliced_skills)
    prepare_skill_dir(skill_type)
    instructor.save_as_readable_json(instructed_skills, f"{ROOT_PATH}/skills/instructed_skills/json/{task_name}.json")
    instructor.save(instructed_skills, f"{ROOT_PATH}/skills/instructed_skills/pkl/{task_name}.pkl")

    # # summarize rough skills
    # summarizer = Summarizer()
    # summaries = summarizer.sliding_summarize(trajectory=trajectory, window_size=5)
    # summarizer.save_as_readable_json(summaries, f"{ROOT_PATH}/skills/rough_skills/json/{task_name}.json")
    # summarizer.save(summaries, f"{ROOT_PATH}/skills/rough_skills/pkl/{task_name}.pkl")

    # locate the skills with only sub-goal
    skill_type = "rlocated_skills"
    skills = pickle.load(open(f"{ROOT_PATH}/skills/sliced_skills/pkl/{task_name}.pkl", "rb"))
    rlocator = RLocator()
    located_skills = rlocator.locate(skills)
    prepare_skill_dir(skill_type)
    rlocator.save_as_readable_json(located_skills, f"{ROOT_PATH}/skills/rlocated_skills/json/{task_name}.json")
    rlocator.save(located_skills, f"{ROOT_PATH}/skills/rlocated_skills/pkl/{task_name}.pkl")

    # # locate the skills
    # # MOD: rough skills -> sliced skills
    # skills = pickle.load(open(f"{ROOT_PATH}/skills/rough_skills/pkl/{task_name}.pkl", "rb"))
    # locator = Locator()
    # located_skills = locator.locate(skills)
    # locator.save_as_readable_json(located_skills, f"{ROOT_PATH}/skills/located_skills/json/{task_name}.json")
    # locator.save(located_skills, f"{ROOT_PATH}/skills/located_skills/pkl/{task_name}.pkl")

    # abstract general skills
    # abstractor = Abstractor()
    # abstracts = abstractor.abstract_skill(summaries)
    # abstractor.save_as_readable_json(abstracts, f"{ROOT_PATH}/skills/general_skills/json/{task_name}.json")
    # abstractor.save(abstracts, f"{ROOT_PATH}/skills/general_skills/pkl/{task_name}.pkl")

    # verify general skills

def generate_exploration_task_from_skill(skill, task_id: int):

    # get the first step url as the url for exploration task
    start_url = skill["sliced_trajectory"][0]["obs"]["url"]
    goal = skill["sub-goal"]
    website = get_website_name_from_url(start_url)

    instruction = skill["instruction"]

    # generate the exploration task
    task = {
      "sites": [
        website
      ],
      "task_id": task_id,
      "require_login": True,
      "storage_state": f"./.auth/{website}_state.json",
      "start_url": f"{start_url}",
      "geolocation": None,
      "intent": f"{goal}",
      "require_reset": False,
    }

    return task

# skill_learning("/home/ubuntu/github/AgentLab/src/agentlab/autogen_policy/annotations/2024-09-01_17-02-43_annotate/2024-09-01_17-02-45_HumanAnnotator_on_webarena.27_51_22d7da")
# skill_learning("/home/ubuntu/agentlab_results/2024-09-11_02-01-51_baseline/2024-09-11_02-01-52_GenericAgent_on_webarena.13_2_ab01c0")



# '''
# generate exploration task from skill
# '''

# # load the skill
# traj_path = "/home/ubuntu/github/AgentLab/src/agentlab/autogen_policy/annotations/2024-09-01_17-02-43_annotate/2024-09-01_17-02-45_HumanAnnotator_on_webarena.27_51_22d7da"
# task_name = traj_path.split("/")[-1]
# skill_path = f"{ROOT_PATH}/skills/located_skills/pkl/{task_name}.pkl"
# skills = pickle.load(open(skill_path, "rb"))
# skill = skills[2]

# # load the tasks
# sim_tasks_path = "/home/ubuntu/github/AgentLab/webarena/simtest.raw.json"
# with open(sim_tasks_path, "r") as f:
#     sim_tasks = json.load(f)

# # choose the largest task_id + 1 as the new task_id
# if len(sim_tasks) == 0:
#     new_task_id = 0
# else:
#     new_task_id = max([task["task_id"] for task in sim_tasks]) + 1

# task = generate_exploration_task_from_skill(skill, new_task_id)

# sim_tasks.append(task)

# with open(sim_tasks_path, "w") as f:
#     json.dump(sim_tasks, f, indent=4)

messages = [
    {
        "role": "user",
        "content": "how many rs are in 'strawberry'?"
    }
]

response = generate_from_4o_chat_completion(messages=messages, model="o1-preview-2024-09-12", temperature=1)
print(response)