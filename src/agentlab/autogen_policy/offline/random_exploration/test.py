from agentlab.autogen_policy.offline.random_exploration.autogen import get_experience_from_exploration, state_summaries_from_exploration
import json
import pickle
from tqdm import tqdm
path = "/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/explorations/2024-08-15_02-37-43_explore/2024-08-15_02-37-45_Explorer_on_webarena.27_92_9938ad"
window_size = 10
step_size = 5
experience = get_experience_from_exploration(path, window_size, step_size)

# # rewriting mode
# with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences/experiences.pkl", "wb") as f:
#     pickle.dump(experience, f)

# # write in json format to visualize
# # only keep url and axtree_txt in obs
# for exp in experience:
#     exp["obs"] = {
#         "url": exp["obs"]["url"],
#     }
#     exp["processed_obs"] = {
#         "axtree_txt": exp["processed_obs"]["axtree_txt"]
#     }

# with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences_json/experiences.json", "w") as f:
#     json.dump(experience, f, indent=4)

############################################################################################################

# appending mode

with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences/new_experiences.pkl", "rb") as f:
    old_experience = pickle.load(f)
    new_experience = old_experience + experience

# save the appended experience
with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences/new_experiences.pkl", "wb") as f:
    pickle.dump(new_experience, f)

# write in json format to visualize
# only keep url and axtree_txt in obs
for exp in experience:
    exp["obs"] = {
        "url": exp["obs"]["url"],
    }
    exp["processed_obs"] = {
        "axtree_txt": exp["processed_obs"]["axtree_txt"]
    }

# append to the json file
with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences_json/new_experiences.json", "r") as f:
    old_experience_json = json.load(f)
    new_experience_json = old_experience_json + experience

with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/experiences_json/new_experiences.json", "w") as f:
    json.dump(new_experience_json, f, indent=4)

# state_summaries = state_summaries_from_exploration(path, "hybrid")


# for state_summary in tqdm(state_summaries):
#     # remove obs from the state_summary dict
#     state_summary.pop("obs")

#     # only keep axtree_txt in processed_obs
    # state_summary["processed_obs"] = {
    #     "axtree_txt": state_summary["processed_obs"]["axtree_txt"]
    # }    
    
# with open("/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/offline/random_exploration/state_summaries.json", "w") as f:
#     json.dump(state_summaries, f, indent=4)