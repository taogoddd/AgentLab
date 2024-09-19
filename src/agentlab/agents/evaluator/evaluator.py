from browsergym.experiments.agent import Agent

class Evaluator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluator = None

    def set_evaluator(self, evaluator):
        self._evaluator = evaluator
