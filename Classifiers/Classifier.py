from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, config):
        self.config = config

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
    def preprocess(self, data):
        pass