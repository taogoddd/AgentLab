from browsergym.experiments.agent import Agent

class Executor(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = None

    def set_executor(self, executor):
        self._executor = executor