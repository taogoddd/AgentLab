import json
from pathlib import Path

def get_avg_score(path: str):
    # get all subdirectories
    subdirs = [x for x in Path(path).iterdir() if x.is_dir()]

    scores = []
    # parse out the id from the subdirectory name, e.g. get 0 from 2024-06-27_11-41-13_GenericAgent_on_webarena.0_51_14d4f1
    for subdir in subdirs:
        subdir_name = subdir.name
        task_name = subdir_name.split("_")[-3]
        id = int(task_name.split(".")[1])
        with open(subdir / "summary_info.json", "r") as f:
            summary_info = json.load(f)
        score = summary_info["cum_reward"]
        scores.append((id, score))
    
    # for subdir in subdirs:
    #     with open(subdir / "summary_info.json", "r") as f:
    #         summary_info = json.load(f)
    #     score = summary_info["cum_reward"]
    #     scores.append(score)
    
    avg_score = sum([score for id, score in scores]) / len(scores)
    print(len(scores))
    print(f"Average score: {avg_score}")

    # get all indexes for avg_score = 1
    successful_ids = [id for id, score in scores if score == 1]
    # sort the ids
    successful_ids.sort()
    # stringfy the ids
    successful_ids = [(id) for id in successful_ids]
    print(f"Number of successful ids: {len(successful_ids)}")
    print(f"Successful ids: {successful_ids}")
    return avg_score

get_avg_score("/home/ytliu/agentlab_results/agentlab_baseline")