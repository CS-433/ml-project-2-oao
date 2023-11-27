from Classifiers.Classifier import Classifier

import numpy as np


class XLNet(Classifier):
    def __init__(self, config: dict, model_path: str = None):
        super().__init__(config, model_path)
        

    def load_model(self, pat: str) -> int:
        raise NotImplementedError

    def train(self, X: np.array, y: np.array):
        raise NotImplementedError

    def predict(self, X: np.array) -> np.array:
        raise NotImplementedError

    def save(self, path: str) -> int:
        raise NotImplementedError