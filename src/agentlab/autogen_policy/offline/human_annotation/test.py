from agentlab.autogen_policy.offline.human_annotation.autogen import *
import numpy as np
from typing import Any
from agentlab.autogen_policy.offline.human_annotation.utils import Obs, ProcessedObs, TrajectoryStep, simplify_readable_trajectory
ROOT_DIR = "/home/ubuntu/github/AgentLab/src/agentlab/autogen_policy"

exp_name = "2024-09-01_17-02-43_annotate"
task_name = "2024-09-01_17-02-45_HumanAnnotator_on_webarena.27_51_22d7da"

web_name = "Reddit"

from typing import TypedDict, Dict, Union, List

'''
generate experiences from annotation
'''
trajectory = get_trajectory_from_annotation(f"{ROOT_DIR}/annotations/{exp_name}/{task_name}")
window_size = 5
res_list = []
readable_res_list = []
# do as a sliding window, with step size of 1
for i in range(0, len(trajectory) - window_size + 1):
    window = trajectory[i:i+window_size]
    res = propose_action_from_annotation(window)
    parsed_res_list = parse(input_string=res, tags=["sub-goal", "instruction", "starting-step", "ending-step",])
    web_name = get_website_name_from_trajectory(window)
    for r in parsed_res_list:
        r["website"] = web_name
    for r in parsed_res_list:
        starting_step = int(r["starting-step"])
        ending_step = int(r["ending-step"])
        sliced_window = window[starting_step:ending_step+1]
        simplified_trajectory = simplify_readable_trajectory(sliced_window)
        r["sliced_trajectory"] = simplified_trajectory
        r["simplified_sliced_trajectory"] = simplified_trajectory
    res_list.extend(parsed_res_list)
# write the readable results to a file as json
# use the task name as the id
with open(f"{ROOT_DIR}/offline/human_annotation/experiences_json/experiences_from_{task_name}.json", "w") as f:
    # do not write the sliced trajectory since not supported by json
    readable_res_list = []
    for res in res_list:
        readable_res = res.copy()
        readable_res.pop("sliced_trajectory")
        readable_res.pop("simplified_sliced_trajectory")
        readable_res_list.append(readable_res)
    json.dump(readable_res_list, f, indent=4)

# write the whole readable results to a file as pickle
with open(f"{ROOT_DIR}/offline/human_annotation/experiences/experiences_from_{task_name}.pkl", "wb") as f:
    pickle.dump(res_list, f)

'''
read the experiences and abstract the content
'''
# with open(f"{ROOT_DIR}/offline/human_annotation/experiences_json/experiences_from_{task_name}.json", "r") as f:
#     experiences = json.load(f)
# print(experiences)
# res_list = abstract_subgoal(experiences)
# print(res_list)
# parsed_res_list = []
# for res in res_list:
#     parsed_res_list = parse(input_string=res, tags=["<generalized sub-goal", "<generalized instruction"])
#     for r in parsed_res_list:
#         r["website"] = web_name
#     res_list.extend(parsed_res_list)
# # write the abstracted subgoals to a file as json
# # use the task name as the id

# with open(f"{ROOT_DIR}/offline/human_annotation/experiences_json/abstracted_experiences_from_{task_name}.json", "w") as f:
#     json.dump(parsed_res_list, f, indent=4)