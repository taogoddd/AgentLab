## Install agentlab

:warning: skip this section if you've already installed the `agentlab` conda env.

install the package locall with the `-e` flag
From the same directory as this README file run:

```bash
    pip install -e .
```

or `conda env update` from the main agentlab directory.

## Launch experiments

For WebArena experiments, you need to setup the websites first(if you have already setup it before, reset it to clear all changes applied by the last experiment!) and then set the env variables:
You may change the urls as your setting
```bash
export SHOPPING="http://localhost:7770"
export SHOPPING_ADMIN="http://localhost:7780/admin"
export REDDIT="http://localhost:9999"
export GITLAB="http://localhost:8023"
export MAP="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
export WIKIPEDIA="http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://localhost:4399"
```

Then also set your API KEY if you are using OpenAI API

Open and modify `exp_configs.py` and `launch_command.py` to your needs. They are
located in `agentlab/experiments/`.

Then set correct PYTHONPATH
```bash
export PYTHONPATH=/path/to/AgentLab
```

Then launch the experiment with

```bash
    python launch_command.py
```

Avoid pushing these changes to the repo unless they are structural changes.
If you prefer launching with command line, see section [Launch experiments with command line](#launch-experiments-with-command-line).

### Debugging jobs

If you launch via VSCode in debug mode, debugging will be enabled and errors will be raised
instead of being logged, unless you set `enable_debug = False` in `ExpArgs`. This
will bring a breakpoint on the error.

To make sure you get a breakpoint at the origin of the error, use the flag
`use_threads_instead_of_processes=True` in `main()` from `launch_exp.py` (or set `n_jobs=1`).


### `joblib`'s parallel jobs
Jobs are launched in parallel using joblib. This will launch multiple processes
on a single computer. The choice is based on the fact that, in general, we are not CPU
bounded. If it becomes the bottleneck we can launch using multiple servers.

SSH to a server with many cores to get more parallelism. You can use `screen` to
ensure the process keeps running even if you disconnect.

```bash
    screen -S <screen_name>
    python launch_command.py
    # ctrl-a d to detach
    # screen -r <screen_name> to reattach
```

## Visualize results
Firstly install jupyter notebook if you haven't already:

```bash
    conda install jupyter
```

Then launch jupyter notebook:
```bash
    jupyter notebook
```

Open `agentlab/experiments/inspect_results.ipynb` in jupyter notebook.

Set your `result_dir` to the right value and run the notebook.



## Launch experiments with command line
Alternatively, you can launch experiments from the command line.

Choose or configure your experiment in `agentlab/experiments/exp_configs.py`.
Make sure it is in the EXP_GROUPS global variable.

Then launch the experiment with

```bash
    python src/agentlab/experiments/launch_exp.py  \
        --savedir_base=<directory/to/save/experiments> \
        --exp_group_name=<name_of_exp_group> \
        --n_jobs=<joblib_pool_size>
```

For example, this will launch a quick test in the default directory:

```bash
    python src/agentlab/experiments/launch_exp.py  \
        --exp_group_name=generic_agent_test \
        --n_jobs=1
```

Some flags are not yet available in the command line. Feel free to add them to
match the interace of main() in `launch_exp.py`.

If you want to test the pipeline of serving OSS LLMs with TGI on Toolkit for evaluation purposes, use `exp_group_name=test_OSS_toolkit` 


## Misc

if you want to download HF models more quickly
```
pip install hf-transfer
pip install torch
export HF_HUB_ENABLE_HF_TRANSFER=1
```

# ACI Dev

The following instructions are for ACI experiments only.

## RESET ENVIRONMENT

YOU MUST RESET THE ENVIRONMENT BEFORE RUNNING LARGE SCALE EXPERIMENTS. the resetting scripts are in src/agentlab/experiments/webarena_scripts/docker_reset.sh. Modify the urls of the websites as your setting.

## For debugging and small tests

You may only run 1 or 2 tasks for debugging. To modify the tasks you will run, go to browsergym/webarena/config.py and change the TASK_IDS variable. To make the results valid, you need to reset the environment each time you run the same tasks that has been run before. However, if you can ensure that the task doesn't modify any data on the website, you may skip this step. BUT THIS ONLY APPLIES FOR SMALL TESTS!

## Augment action space

Actions outputted by the model are in the format of a function call. e.g. click(123). The definition of the action functions are in core/action/functions.py.

Here are general steps to augment the action space

1. Go to browsergym/core/action/functions.py. Define your action function here. 
    - If your action needs to call a external function like what "send_msg_to_user" does, you may define the external functions you need as None firstly and in the steps later, we will instruct you to define the external functions and pass it through.
    - Add proper comments as other functions do. This will be prompted to the agent as the description of the function. 
    - IMPORTANT: you should always add [AUGMENT] before the description to distinguish them from the original functions. The prompt to the LLM will be based on this.

2. [Optional] If your function contains external functions (e.g. send_message_to_user) that have not been defined yet. Go to browsergym/core/env.py and navigate to the step function in the BrowserGym class. defines the external functions here (e.g. send_message_to_user) and pass it through the execute_python_code.

3. Go to browsergym/core/actions/highlevel.py.
    - import the action function from .functions
    - group the actions and add the actions like the following example:
    ```python
    SHOPPING_ADMIN_EXTRA_ACTIONS = [go_to_reviews_page]
    ```
    - in the class HighLevelActionSet, add the action_group name(will be passed through args) into ActionSubset and the corresponding part in __init__.py like the following example:
    ```python
    def __init__():
    ...
    if subsets:
            for subset in subsets:
                match subset:
                    case "chat":
                        allowed_actions.extend(CHAT_ACTIONS)
                    case "infeas":
                        allowed_actions.extend(INFEAS_ACTIONS)
                    ...
                    # add a new group
                    case "shopping":
                        allowed_actions.extend(SHOPPING_ADMIN_EXTRA_ACTIONS)
    ```
4. Change your experiment args correspondingly to include the augmented action space. 
    - Go to src/agentlab/experiments/exp_configs.py and find function aci_study. All configs are defined here.
    - Since we augment the action space with the action set "shopping", we need to append this to our previous action_set. The default action_set is "bid". Now you may change it to "bid+shopping" in the args of the experiment. If you have further action sets, you may append them with a "+".
    - For example, if you have a new action set "shopping_admin", you may change the action_set to "bid+shopping+shopping_admin".
Now the action should work smoothly!

# Notes

1. browsergym and webarena packages are moved inside AgentLab codes for easier development; Due to space limitation of Github, browsergym/workarena/data_files are gitignored. Please get it from the original package if needed.