class Task:
    """ Abstract class for cognitive tasks.
    Meant to be implemented by the task in this directory

    """
    default_params = None

    def __init__(self, params):
        self.__dict__.update(self.default_params)
        self.__dict__.update(params)

    def build_train_batch(self):
        pass

    def set_params(self, **kwargs):
        params = self.default_params.copy()
        for key, val in kwargs:
            params[key] = val
        return params

    def generate_train_trials(self):
        while 1 > 0:
            yield self.build_train_batch()
