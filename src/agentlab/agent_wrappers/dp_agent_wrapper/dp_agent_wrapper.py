from collections import deque
from dataclasses import dataclass
from agentlab.agents.ap_agent.ap_agent import AutoGenPolicyAgentArgs, AutoGenPolicyAgent
from typing import Deque, List
import re
from agentlab.agents.planner.planner import Planner
from agentlab.agents.evaluator.evaluator import Evaluator
from agentlab.agents.executor.executor import Executor

@dataclass
class HistoryStep:
    plan: str
    trajectory: list
    eval: dict

class DPAgentWrapper:
    history = []
    def __init__(self, planner: Planner, evaluator: Evaluator, executor: Executor):
        self.planner = planner
        self.evaluator = evaluator
        self.executor = executor
        self.history: List[HistoryStep] = []
    
    def is_done(self, action):
        # check if the action is a stop action
        if "stop" in action:
            return True
        return False
    
    def get_action(self, obs):
        history = self.history
        if history == []:
            self.history.append(HistoryStep(plan=None, trajectory=[{"obs": obs}], eval={"done": False}))

        # get plan for current step if not already present
        current_step = self.history[-1]
        if current_step.plan is None:
            plan = self.planner.get_plan(obs)
            current_step.plan = plan
        
        while not current_step["eval"]["done"]:
            action = self.executor.get_action(history)
            current_step[-1]["action"] = action
            eval = self.evaluator.get_eval(history)
            current_step["history"][-1]["eval"] = eval




        

        
        
