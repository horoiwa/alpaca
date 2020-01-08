

class BaseOptimizer:

    def __init__(self):

        self.objectives = {}
        self.models = {}

    def add_X(self, X, adapter_config):
        pass

    def add_objective(self, obj_name, model, fit):
        pass

    def add_constraints(self, key, variables):
        pass

    def setup(self):
        pass

    def optimize(self):
        pass

    def _validation(self, X, y):
        pass


class GAOptimizer(BaseOptimizer):
    pass
    def __init__(self):
        pass
