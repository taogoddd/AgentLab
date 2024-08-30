from collections import deque
from agentlab.agents.ap_agent.ap_agent import AutoGenPolicyAgentArgs, AutoGenPolicyAgent
from typing import Deque
import re

class AgentManager:
    agents: Deque[AutoGenPolicyAgent]
    def __init__(self, root_agent_name: str):
        self.agents = deque()
        root_agent_args = AutoGenPolicyAgentArgs(agent_name=root_agent_name)
        self.agents.appendleft(root_agent_args.make_agent(root_agent_name))
    
    def add_agent(self, agent_name: str):
        agent_args = AutoGenPolicyAgentArgs(agent_name=agent_name)
        self.agents.appendleft(agent_args.make_agent(agent_name))

    def is_low_level_action(self, action):
        action_type = action.split()[0]
        return (action_type in self.low_level_action_list)

    def is_high_level_action(self, action):
        action_type = action.split()[0]
        return (action_type in self.action_to_prompt_dict)
    
    def is_done(self, action):
        if "stop" in action:
            return True
        return False
    
    def get_action(self, obs):
        action, thinking = None, None
        while self.agents:
            # get current working agent
            agent = self.agents[0]
            # get action from the agent
            _, action_dict = agent.get_action(obs)
            action = action_dict["action"]
            thinking = action_dict["think"]

            # check whether the action is a stop action
            if self.is_done(action):
                self.agents.popleft()
                if not self.agents:
                    # get the top agent
                    self.agents[0].add_response(re.search(r"\[(.*?)\]", action).group(1))
                continue

            # the action is not a stop action
            if self.is_low_level_action(action):
                # no response for low level actions (temporarily)
                agent.add_response("")
                return action, thinking
            if self.is_high_level_action(action):
                # parse the action
                pattern = r'(\w+)\s+\((.*?)\)'
                matches = re.findall(pattern, action)
                action_name, _ = matches[0]

                # add the agent
                self.add_agent(action_name)
                continue
        return action, action_dict

        
        
