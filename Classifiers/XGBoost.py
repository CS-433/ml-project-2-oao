from Classifiers.Classifier import Classifier
import numpy as np
import xgboost as xgb
import os


class XGBoost(Classifier):
    def __init__(self, config: dict, model_path: str = None):
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str) -> int:
        if os.path.exists(path):
            self.model = xgb.Booster()
            self.model.load_model(path)
            return 0  # Model loaded successfully
        else:
            return -1  # Model file not found

    def train(self, X: np.array, y: np.array):
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.config, dtrain)

    def predict(self, X: np.array) -> np.array:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path: str) -> int:
        if self.model:
            self.model.save_model(path)
            return 0  # Model saved successfully
        else:
            return -1  # No model to save