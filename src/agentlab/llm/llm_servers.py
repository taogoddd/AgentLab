from dataclasses import fields
from datetime import datetime
import json
import os

from agentlab.llm.chat_api import ChatModelArgs
from agentlab.llm.llm_configs import (
    CHAT_MODEL_ARGS_DICT,
    CONTEXT_WINDOW_EXTRA_GPU,
    INFRA_HPARAMS_DICT_BASE,
    CLOSED_SOURCE_APIS,
)
from typing import List
import logging
import subprocess
import time

from huggingface_hub import InferenceClient
import yaml

from agentlab.llm import toolkit_configs


class LLMServers:
    def __init__(self, exp_args_list=()) -> None:
        self.server_dict = {}
        self.start_all_servers(exp_args_list)

    def close_all_servers(self):
        for _, server_info in self.server_dict.items():
            job_id = server_info["job_id"]
            kill_server(job_id)

    def start_all_servers(self, exp_args_list):
        # type: (List[ChatModelArgs]) -> None
        """Launch the unique set of required servers for all experiments in exp_args_list."""
        for exp_args in exp_args_list:
            self.start_llm_servers_for_agent(exp_args.agent_args)

    def start_llm_servers_for_agent(self, agent_args):
        """Launch a server and set the url in agent_kwargs inplace."""
        # TODO agent_args should implement get_chat_models_args, returning a
        # list of all chat models, instead of doing introspection.
        for arg in fields(agent_args):
            arg = getattr(agent_args, arg.name)
            if isinstance(arg, ChatModelArgs):
                self.get_url(arg)

    def get_url(self, chat_model_args: ChatModelArgs):
        """Get the url of the server with the given kwargs. If it doesn't exist, launch a new server."""
        if chat_model_args.model_url is not None:
            return
        if any(chat_model_args.model_name.startswith(api) for api in CLOSED_SOURCE_APIS):
            return
        if chat_model_args.key() in self.server_dict:
            chat_model_args.model_url = self.server_dict[chat_model_args.key()]["model_url"]
            return
        chat_model_key = chat_model_args.key()
        if chat_model_key not in self.server_dict:
            self.server_dict[chat_model_key] = {}
            job_id, model_url = auto_launch_server(chat_model_args)
            self.server_dict[chat_model_key]["model_url"] = model_url
            self.server_dict[chat_model_key]["job_id"] = job_id
            self.server_dict[chat_model_key]["is_ready"] = False
            chat_model_args.model_url = model_url
        return

    def wait_for_server(self, chat_model_key):
        """Wait for all servers to be ready."""

        if chat_model_key not in self.server_dict:
            return

        logging.warning(
            f"if n_jobs < # experiments, it could lead to deadlock. make sure shuffle_jobs = False"
        )

        server_info = self.server_dict[chat_model_key]
        job_id = server_info["job_id"]
        model_url = server_info["model_url"]
        is_ready = server_info["is_ready"]
        while not is_ready:
            time.sleep(3)
            is_ready = check_server_status(job_id, model_url)
        self.server_dict[chat_model_key]["is_ready"] = is_ready


def launch_toolkit_tgi_server(
    job_name: str = "tgi_custom",
    model_name: str = None,
    max_total_tokens: int = 16_384,
    max_new_tokens: int = 256,
    gpu: int = 1,
    gpu_mem: int = 80,
    cpu: int = 4,
    mem: int = 64,
    n_shard: int = 1,
    max_run_time: int = 172_800,
    extra_tgi_args: dict = None,
    sliding_window: bool = False,  # TODO implement
):
    if model_name is None:
        raise ValueError("Model name must be provided.")

    max_input_length = max_total_tokens - max_new_tokens
    now = datetime.now()
    job_name_now = f"{job_name}_{now.strftime('%y%m%d_%H%M%S')}"

    # NOTE: you need to set MAX_BATCH_PREFILL_TOKENS >= MAX_INPUT_LENGTH
    max_batch_prefill_tokens = max_input_length + 50  # NOTE: TGI suggestion
    # NOTE: you need to set MAX_BATCH_TOTAL_TOKENS >= MAX_BATCH_PREFILL_TOKENS
    # NOTE: MAX_BATCH_TOTAL_TOKENS is inferred but still need to be set if we want TGI to output the infered value
    max_batch_total_tokens = int(2 * max_batch_prefill_tokens)

    # tgi_image = toolkit_configs.TGI_IMAGE_LLMD
    tgi_image = toolkit_configs.TGI_IMAGE_OFFICIAL

    if model_name.startswith("/"):
        assert model_name.startswith(toolkit_configs.UI_COPILOT_DATA_PATH)

    # Build the command using configurations and provided arguments
    # TODO: clean
    launch_command = f"""
        eai job new \\
            --name {job_name_now} \\
            --account {toolkit_configs.ACCOUNT_NAME} \\
            --image {tgi_image} \\
            --gpu {gpu} \\
            --gpu-mem {gpu_mem} \\
            --cpu {cpu} \\
            --mem {mem} \\
            --max-run-time {max_run_time} \\
            --data {toolkit_configs.UI_COPILOT_DATA_VOLUME}:{toolkit_configs.UI_COPILOT_DATA_PATH} \\
            --data {toolkit_configs.RESULTS_VOLUME}:{toolkit_configs.RESULTS_PATH} \\
            --data {toolkit_configs.UI_COPILOT_DATA_VOLUME}:/data \\
            --env HUGGINGFACE_HUB_CACHE=/data/huggingface_hub_cache \\
            --env FLASH_ATTENTION=1 \\
            --env MODEL_ID={model_name} \\
            --env MAX_INPUT_LENGTH={max_input_length} \\
            --env MAX_BATCH_PREFILL_TOKENS={max_batch_prefill_tokens} \\
            --env MAX_TOTAL_TOKENS={max_total_tokens} \\
            --env MAX_BATCH_TOTAL_TOKENS={max_batch_total_tokens} \\
            --env NUM_SHARD={n_shard} \\
            --env PORT=8080 \\
            --env HUGGING_FACE_HUB_TOKEN={os.environ["HUGGINGFACEHUB_API_TOKEN"]} \\
    """
    if extra_tgi_args:
        for key, value in extra_tgi_args.items():
            launch_command += f"--env {key}={value} \\\n"
    launch_command += "--restartable"

    # TODO: figure out why LLMD was using the following env var
    # --data {toolkit_configs.LLMD_TRANSFORMERS_CACHE}:/data \\
    # --env HOME=/home/toolkit \\
    # TODO: do we want to add support for this:
    # --env ROPE_SCALING=dynamic \\
    # --env ROPE_FACTOR=2.0 \\
    logging.info("Launching server with command:")
    logging.info(launch_command)

    result = run_subprocess(launch_command)
    time.sleep(1)

    # Extract job ID and URL
    get_id_command = "eai job info --last"
    result = run_subprocess(get_id_command)
    stdout = result.stdout.strip()
    job_data = yaml.safe_load(stdout)
    job_id = job_data["id"]
    model_url = f"https://{job_id}.job.console.elementai.com"

    # Grant permission
    ROLE_NAME = os.environ["ROLE_NAME"]
    assert ROLE_NAME, "you must specify a role name in your env variables"
    grant_permission_command = f"eai role policy new {toolkit_configs.ACCOUNT_NAME}.{ROLE_NAME} job:get@$(eai job get ${{JOB_ID}} --field urn)"
    run_subprocess(grant_permission_command)

    return job_id, model_url


def auto_launch_server(chat_model_args: ChatModelArgs) -> str:
    """Launch a server with the given kwargs. Return the url of the server.

    Parameters
    ----------
    chat_model_args : ChatModelArgs
        An object that can instantiate a chat model.
    """

    model_path = (
        chat_model_args.model_path if chat_model_args.model_path else chat_model_args.model_name
    )
    # TODO: should max_total_tokens be overriden?
    max_total_tokens = chat_model_args.max_total_tokens
    model_size = chat_model_args.model_size
    total_params = compute_total_params(model_size)

    # Find the ceiling of total_params
    ceiling = next(key for key in INFRA_HPARAMS_DICT_BASE.keys() if key >= total_params)

    gpu = INFRA_HPARAMS_DICT_BASE[ceiling].get("gpu")
    cpu = INFRA_HPARAMS_DICT_BASE[ceiling].get("cpu")
    mem = INFRA_HPARAMS_DICT_BASE[ceiling].get("mem")
    gpu_mem = INFRA_HPARAMS_DICT_BASE[ceiling].get("gpu_mem", 80)
    extra_tgi_args = chat_model_args.extra_tgi_args

    # adjust gpu based on context window
    gpu += CONTEXT_WINDOW_EXTRA_GPU.get(max_total_tokens, 0)

    if chat_model_args.shard_support:
        # NOTE: n_shard needs to be a power of 2 to properly shard the n_heads and activations
        n_shard = 2 ** (gpu.bit_length() - 1)
    else:
        n_shard = 1

    job_id, model_url = launch_toolkit_tgi_server(
        job_name="ui_copilot_tgi_server",
        model_name=model_path,
        max_total_tokens=max_total_tokens,
        gpu=gpu,
        cpu=cpu,
        mem=mem,
        gpu_mem=gpu_mem,
        n_shard=n_shard,
        extra_tgi_args=extra_tgi_args,
    )

    return job_id, model_url


## Utility functions
def run_subprocess(command):
    """Utility function to run subprocess commands with error handling."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while executing command: {e}")
        logging.error(e.output)  # Print the error output
        raise


def compute_total_params(value):
    """Compute the total number of parameters for MoE models of the form 'n_experts x expert_size'."""
    if isinstance(value, str) and "x" in value:
        # Split the string by 'x' and convert the parts to integers, then multiply
        factors = value.split("x")
        return int(factors[0]) * int(factors[1])
    # if int or float, return as is
    elif isinstance(value, (int, float)):
        return value
    else:
        raise ValueError("Unsupported value format")


def check_server_status(job_id: str, model_url: str) -> bool:
    """
    Checks the server status at the specified URL.

    Parameters:
    url (str): The URL of the server to be checked.

    Returns:
    bool: True if the server is ready, False otherwise.
    """

    # first check if toolkit job is still running
    command = f"eai job info {job_id}"
    try:
        result = run_subprocess(command)
    except:
        raise Exception(f"tell Massimo if you hit this bug. it's sporadic and I can't reproduce")

    stdout = result.stdout.strip()
    data = yaml.safe_load(stdout)

    job_status = data.get("state", {})

    if job_status in ["QUEUING", "QUEUED"]:
        logging.info(f"Toolkit job {job_id} is still {job_status}")
        return False
    if job_status == "RUNNING":
        client = InferenceClient(model=model_url, token=os.environ["TGI_TOKEN"])
        try:
            client.text_generation(prompt="hello")
            logging.info(f"TGI server for job {job_id} is ready")
            return True
        except:
            logging.info(f"Waiting for job {job_id}'s TGI server to be ready...")
            return False
    else:
        raise Exception(f"Toolkit job {job_id} is {job_status}")


def kill_server(job_id: str):
    """Kill the server at the given url."""

    command = f"eai job kill {job_id}"

    run_subprocess(command)

    logging.info(f"submitted kill command for Toolkit job {job_id}...")

    time.sleep(1)
    command = f"eai job info {job_id}"
    result = run_subprocess(command)

    stdout = result.stdout.strip()
    data = yaml.safe_load(stdout)
    job_status = data.get("state", {})
    if job_status not in ["CANCELLED", "CANCELLING", "FAILED"]:
        logging.info(f"Toolkit job {job_id} is still {job_status}")


def kill_all_servers(job_name="ui_copilot_tgi_server"):
    """Kill all servers that have been left running unintentionally."""

    # get all the job_id of the running jobs
    command = f"eai job ls --reverse --format json"
    result = run_subprocess(command)

    stdout = result.stdout.strip()
    json_strings = stdout.split("\n")
    jobs = [json.loads(json_str) for json_str in json_strings]
    for job in jobs:
        if job_name in job["name"]:
            if job["state"] in ["RUNNING", "QUEUED", "QUEUING"]:
                kill_server(job["id"])


if __name__ == "__main__":

    ## launching a TGI server on Toolkit

    # model = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model = "finetuning/Meta-Llama-3-8B-Instruct"
    # model = "deepseek-ai/DeepSeek-V2-Chat"
    # model = "microsoft/WizardLM-2-8x22B"
    # model = "bigcode/starcoderplus"
    # model = "codellama/CodeLlama-34b-Python-hf"
    # model = "codellama/CodeLlama-13b-Python-hf"
    # model = "codellama/CodeLlama-7b-Python-hf"
    model = "microsoft/Phi-3-mini-128k-instruct"

    # auto_launch_server(CHAT_MODEL_ARGS_DICT[model])

    kill_all_servers()
