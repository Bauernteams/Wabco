class BaseTrain:
    def __init__(self, model, data, config):
        self.model = model.model
        self.modelType = model.modelType
        self.data = data
        self.config = config

    def train(self):
        raise NotImplementedError
