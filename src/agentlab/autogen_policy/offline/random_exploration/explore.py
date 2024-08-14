# from browsergym.experiments.loop import EnvArgs, ExpArgs
# from browsergym.experiments.agent import Agent
# from agentlab.experiments import task_collections as tasks
# from agentlab.experiments import args
# from browsergym.core.action.python import PythonActionSet
# from agentlab.agents import dynamic_prompting as dp
# import gymnasium as gym
# import logging
# from browsergym.experiments.loop import AbstractAgentArgs

# def annotate(benchmark="webarena"):


# def main():

#     TASK_ID = 0
#     benchmark = "webarena"
#     exp_dir = "/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/trajectories"
#     trajectory = Trajectory()
#     obs = None

#     env_args_list = tasks.get_benchmark_env_args(benchmark, max_steps=30)

#     env_args=args.CrossProd(env_args_list)

#     action=dp.ActionFlags(
#             multi_actions=False,
#             # change action space here
#             action_set="wa_bid+reddit",
#             long_description=True,
#             individual_examples=True,
#         )

#     action_set = dp.make_action_set(action)

#     env: gym.Env = env_args.make_env(
#         action_mapping=action_set.to_python_code, exp_dir=exp_dir
#     )

#     print("Env created")

#     # reset the env
#     obs, env_info = env.reset()

#     # interaction
#     while not trajectory.is_done:
#         action = get_action(obs, )

"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Don't push your changes to this file to git unless you are making structural changes.
"""

from agentlab.analyze.inspect_results import get_most_recent_folder
from agentlab.experiments.launch_exp import main
from agentlab.experiments.exp_utils import RESULTS_DIR

# set basic config of loggig to debug
import logging

logging.basicConfig(level=logging.ERROR)


exp_args_list = None

## select your experiment group here from exp_configs.py
# exp_group_name = "generic_agent_test"  ## this will make a very quick test
# exp_group_name = "generic_agent_eval_llm"
# exp_group_name = "random_search"
# exp_group_name = "ablation_study_GPT_3_5"

## or from exp_configs_OSS.py
# exp_group_name = "tgi_toolkit_test"  ## this will make a very quick test
# exp_group_name = "OSS_random_search"

## you can also specify the experiment group name directly here to relaunch it
# exp_group_name = "2024-01-22_23-46-25_random_search_prompt_OSS_LLMs"

# WorkArena Ablation Study for ICML
# exp_group_name = "2024-02-01_03-20-14_ablation_study_browsergym_workarena"

# MiniWob Ablation Study for ICML
# exp_group_name = "2024-02-01_03-24-01_ablation_study_browsergym_miniwob"

# exp_group_name = get_most_recent_folder(RESULTS_DIR).name

# WebArena ACI Study
exp_group_name = "explore"

# relaunch_mode = "incomplete_only"
# relaunch_mode = "all_errors"
relaunch_mode = None

RESULTS_DIR = "/home/ytliu/github/AgentLab/src/agentlab/autogen_policy/explorations"

main(
    exp_root=RESULTS_DIR,
    benchmark='webarena', # this will be ignored if you specify the exp_group_name
    exp_group_name=exp_group_name,
    model_name='azure-gpt-4o',
    exp_args_list=exp_args_list,
    n_jobs=1,  # 1 for debugging, -1 for all cores except 2
    relaunch_mode=relaunch_mode,  # choices = [None, 'incomplete_only', 'all_errors', 'server_errors']. if not None, make sure you're pointing to an existing experiment directory
    auto_accept=True,  # skip the prompt to accept the experiment
)
