from Classifier import Classifier

import pandas as pd

class BERT(Classifier):
    def __init__(self, config: dict, model_path: str = None):
        super().__init__(config, model_path)


    def train(self, data: pd.DataFrame) -> None:
        """
        trains the model
        :param data: training data
        """
        pass

    def predict(self, data: pd.DataFrame) -> list:
        """
        predicts the labels for the data
        :param data: data to be predicted
        :return: predictions
        """
        pass

    def save(self, path: str) -> None:
        """
        saves the model
        :param path: path to save the model
        """
        pass

    def load(self, path: str) -> None:
        """
        loads the model
        :param path: path to load the model
        """
        pass