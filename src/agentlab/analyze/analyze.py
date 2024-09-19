import json
from pathlib import Path
import pandas as pd

def new_get_avg_score(path: str):
    subdirs = [x for x in Path(path).iterdir() if x.is_dir()]

    records = {
        "all_ids": [],
        "successful_ids": [],
        "failed_ids": [],
    }

    not_completed_ids = []

    scores = []
    # parse out the id from the subdirectory name, e.g. get 0 from 2024-06-27_11-41-13_GenericAgent_on_webarena.0_51_14d4f1
    for subdir in subdirs:
        subdir_name = subdir.name
        id = int(subdir_name.split(".")[1])
        # check whether file exists
        if (subdir / "0" / "summary_info.json").exists():
            with open(subdir / "0" / "summary_info.json", "r") as f:
                summary_info = json.load(f)
            score = summary_info["cum_reward"]
            scores.append((id, score))
        else:
            not_completed_ids.append(id)
    
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

    not_completed_ids.sort()
    print(f"Number of not completed ids: {len(not_completed_ids)}")
    print(f"Not completed ids: {not_completed_ids}")

    records["all_ids"] = [id for id, score in scores]
    records["successful_ids"] = successful_ids
    # only keep completed but failed tasks
    completed_ids = [id for id in records["all_ids"] if id not in not_completed_ids]
    records["failed_ids"] = [id for id in completed_ids if id not in successful_ids]
    
    return avg_score, records

def get_avg_score(path: str):
    # get all subdirectories
    subdirs = [x for x in Path(path).iterdir() if x.is_dir()]

    records = {
        "all_ids": [],
        "successful_ids": [],
        "failed_ids": [],
    }

    not_completed_ids = []

    scores = []
    # parse out the id from the subdirectory name, e.g. get 0 from 2024-06-27_11-41-13_GenericAgent_on_webarena.0_51_14d4f1
    for subdir in subdirs:
        subdir_name = subdir.name
        task_name = subdir_name.split("_")[-3]
        id = int(task_name.split(".")[1])
        # check whether file exists
        if (subdir / "summary_info.json").exists():
            with open(subdir / "summary_info.json", "r") as f:
                summary_info = json.load(f)
            score = summary_info["cum_reward"]
            scores.append((id, score))
        else:
            not_completed_ids.append(id)
    
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

    not_completed_ids.sort()
    print(f"Number of not completed ids: {len(not_completed_ids)}")
    print(f"Not completed ids: {not_completed_ids}")

    records["all_ids"] = [id for id, score in scores]
    records["successful_ids"] = successful_ids
    # only keep completed but failed tasks
    completed_ids = [id for id in records["all_ids"] if id not in not_completed_ids]
    records["failed_ids"] = [id for id in completed_ids if id not in successful_ids]
    
    return avg_score, records

def get_merged_avg_score(path_list: list):
    '''
    Get the merged average score from multiple result directories
    One task is considered successful if it has a score of 1 in any of the result directories
    '''
    all_scores = []
    # init all_successful_ids with paths as the keys
    all_successful_ids = {} 
    all_failed_ids = []
    all_not_completed_ids = []
    for path in path_list:
        if path not in all_successful_ids:
            all_successful_ids[path] = []
        avg_score, records = get_avg_score(path)
        successful_ids = records["successful_ids"]
        for id in successful_ids:
            if id not in all_successful_ids:
                all_successful_ids[path].append(id)
    all_ids = [i for i in range(812)]

    # sort every list of successful ids
    for path, successful_ids in all_successful_ids.items():
        successful_ids.sort()
        all_successful_ids[path] = successful_ids
    merged_successful_ids = set()
    for path, successful_ids in all_successful_ids.items():
        merged_successful_ids.update(successful_ids)
    merged_avg_score = len(merged_successful_ids) / len(all_ids)
    failed_ids = [id for id in all_ids if id not in merged_successful_ids]
    print(f"All successful ids: {merged_successful_ids}")
    print(f"Merged average score: {merged_avg_score}")
    return merged_avg_score, all_successful_ids

def get_merged_avg_template_score(path_list: list):
    _, records = get_avg_score(path_list[0])
    all_ids = records["all_ids"]
    _, all_successful_ids = get_merged_avg_score(path_list)
    task_template_mapping = get_task_template_id_mapping()
    num_templates = len(task_template_mapping)
    successful_template_ids = []
    # get the list of all successful ids
    all_successful_id_set = set()
    for path, successful_ids in all_successful_ids.items():
        all_successful_id_set.update(successful_ids)
    all_successful_id_list = list(all_successful_id_set)
    for id in all_successful_id_list:
        for template_id, ids in task_template_mapping.items():
            if id in ids and template_id not in successful_template_ids:
                successful_template_ids.append(template_id)
    
    # get the list of all tasks of the successful templates
    successful_template_ids_list = list(successful_template_ids)
    successful_template_task_ids = []
    for template_id in successful_template_ids_list:
        successful_template_task_ids.extend(task_template_mapping[template_id])
    
    merged_avg_template_score = len(successful_template_task_ids) / len(all_ids)
    print(f"Number of successful templates: {len(successful_template_ids)}")
    print(f"Successful templates: {successful_template_ids}")
    print(f"Merged average template score: {merged_avg_template_score}")

def get_sub_domain_avg_score(sub_domain: str, results_dir: str):
    sub_domain_ids = get_sub_domain_ids(sub_domain)
    scores = []
    subdirs = [x for x in Path(results_dir).iterdir() if x.is_dir()]
    successful_ids = []
    for subdir in subdirs:
        subdir_name = subdir.name
        task_name = subdir_name.split("_")[-3]
        id = int(task_name.split(".")[1])
        if id in sub_domain_ids:
            # check whether file exists
            if (subdir / "summary_info.json").exists():
                with open(subdir / "summary_info.json", "r") as f:
                    summary_info = json.load(f)
                score = summary_info["cum_reward"]
                scores.append(score)
                if score == 1:
                    successful_ids.append(id)
    successful_ids.sort()
    avg_score = sum(scores) / len(scores)
    print(f"Average score for sub domain {sub_domain}: {avg_score}")
    print(f"Number of successful ids: {len(successful_ids)}")
    print(f"Successful ids: {successful_ids}")

def get_sub_domain_ids(sub_domain: str, include_multi_sites: bool = False):
    '''
    Get all task ids that have the sub_domain in their sites

    Args:
    sub_domain: str - the sub_domain to search for
    include_multi_sites: bool - whether to include tasks with multiple sites
    '''
    sub_domain_ids = []
    with open("/home/ytliu/github/AgentLab/webarena/test.raw.json", "r") as f:
        data = json.load(f)
        for task in data:
            if sub_domain in task["sites"] and (include_multi_sites or len(task["sites"]) == 1):
                sub_domain_ids.append(task["task_id"])
    return sub_domain_ids

def dump_sub_domain_data(sub_domain: str):
    with open("/home/ytliu/github/AgentLab/webarena/test.raw.json", "r") as f:
        data = json.load(f)
        # only keep the task whose sites have reddit
        new_data = [task for task in data if sub_domain in task["sites"]]

    with open(f"/home/ytliu/github/AgentLab/webarena/test.{sub_domain}.json", "w") as f:
        json.dump(new_data, f, indent=4)

def analyze_step(step_result_dir: str = "/home/ytliu/agentlab_results/2405_all_tasks_step_webarena_bugfix"):
    # Load the CSV file
    df = pd.read_csv(f'{step_result_dir}/summary.csv')

    # Filter the DataFrame for rows where reward equals 1
    successful_tasks = df[df['reward'] == 1]

    # Extract the task IDs
    successful_task_ids = successful_tasks['task_id'].tolist()

    # reddit_ids = [27, 28, 29, 30, 31, 66, 67, 68, 69, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 552, 553, 554, 555, 562, 563, 564, 565, 566, 580, 581, 582, 583, 584, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 671, 672, 673, 674, 675, 681, 682, 683, 684, 685, 686, 687, 688, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 791]
    reddit_ids = get_sub_domain_ids("reddit", include_multi_sites=False)
    
    print(f"Reddit ids: {reddit_ids}")
    print(f"Number of reddit tasks: {len(reddit_ids)}")
    # Filter the successful tasks for the Reddit tasks
    successful_reddit_tasks = [task_id for task_id in successful_task_ids if task_id in reddit_ids]

    print(f"Number of successful tasks: {len(successful_task_ids)}")
    print(f"Successful tasks: {successful_task_ids}")

    print(f"Number of successful Reddit tasks: {len(successful_reddit_tasks)}")
    print(f"Successful Reddit tasks: {successful_reddit_tasks}")

def get_task_template_id_mapping():
    with open("/home/ytliu/github/AgentLab/webarena/test.raw.json", "r") as f:
        data = json.load(f)
        task_template_mapping: dict[str, list] = {}
        for task in data:
            task_id = task["task_id"]
            template_id = task["intent_template_id"]
            if template_id not in task_template_mapping:
                task_template_mapping[template_id] = []
            task_template_mapping[template_id].append(task_id)
    return task_template_mapping

# get_avg_score("/home/ytliu/agentlab_results/2024-09-11_02-01-51_baseline")
new_get_avg_score("/home/ytliu/github/AgentLab/results/streaming_single_action_merged_skills_all_dynamics20240919183534")
# get_merged_avg_score(["/home/ytliu/agentlab_results/agentlab_baseline", "/home/ytliu/agentlab_results/2024-09-11_02-01-51_baseline"])
# get_merged_avg_template_score(["/home/ytliu/agentlab_results/agentlab_baseline", "/home/ytliu/agentlab_results/2024-09-11_02-01-51_baseline"])
# get_sub_domain_avg_score("reddit", "/home/ytliu/agentlab_results/2024-08-15_03-45-52_offline_learning")
# print(get_sub_domain_ids("reddit"))
# /home/ytliu/agentlab_results/2024-08-15_03-45-52_offline_learning

# get_sub_domain_avg_score("reddit", "/home/ytliu/agentlab_results/agentlab_baseline")
# analyze_step()
# get_sub_domain_ids("reddit", include_multi_sites=False)

# with open("/home/ytliu/github/AgentLab/src/agentlab/skills/shopping_admin/skills.json", "r") as f:
#     skills = json.load(f)

# for skill in skills:
#     if skill["type"] == "navi" and skill["name"] == None:
#         # reparse the page summary
#         summary = skill["page-summary"]
#         from agentlab.extract_navi import parse_page_summary
#         name, description, usages = parse_page_summary(summary)
#         skill["name"] = name
#         skill["description"] = description
#         skill["usages"] = usages

# with open("/home/ytliu/github/AgentLab/src/agentlab/skills/shopping_admin/skills_1.json", "w") as f:
#     json.dump(skills, f, indent=2)

# with open("/home/ytliu/agentlab_results/agentlab_baseline/2024-06-27_11-41-13_GenericAgent_on_webarena.0_51_14d4f1/exp_args.pkl", "rb") as f:
#     import pickle
#     exp_args = pickle.load(f)
#     print(exp_args)