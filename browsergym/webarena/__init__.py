from browsergym.core.registration import register_task

# register the WebArena benchmark
from .task import GenericWebArenaTask, SimulatedWebArenaTask
from .config import TASK_IDS, SIM_TASK_IDS

ALL_WEBARENA_TASK_IDS = []

ALL_SIM_WEBARENA_TASK_IDS = []

# register the WebArena benchmark
for task_id in TASK_IDS:
    gym_id = f"webarena.{task_id}"
    register_task(
        gym_id,
        GenericWebArenaTask,
        kwargs={"task_kwargs": {"task_id": task_id}},
    )
    ALL_WEBARENA_TASK_IDS.append(gym_id)

# register the sim WebArena benchmark
for task_id in SIM_TASK_IDS:
    gym_id = f"sim_webarena.{task_id}"
    register_task(
        gym_id,
        SimulatedWebArenaTask,
        kwargs={"task_kwargs": {"task_id": task_id}},
    )
    ALL_SIM_WEBARENA_TASK_IDS.append(gym_id)
    print(f"Registered {gym_id}")