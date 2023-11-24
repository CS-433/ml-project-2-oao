from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, config, model_path=None):
        self.config = config
        self.model = None
        if model_path:
            model = self.load_model(model_path)
        self.metrics = {}

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def train(self, train_data, train_labels):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def get_metrics(self):
        return self.metrics.copy()