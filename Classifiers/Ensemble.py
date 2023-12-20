from sklearn.ensemble import VotingClassifier
from Classifiers.BERT import BERT
from Classifiers.XLNet import XLNet
from Classifiers.XGBoost import XGBoost
from Classifiers.Classifier import Classifier
import numpy as np
import pickle

# Assuming you have two custom classifiers

class Ensemble(Classifier):
    def __init__(self, config: dict):
        super().__init__(config)
        if self.model == None:
            self.build_classifier()


    def build_classifier(self):
        xlnet = XLNet(self.config['xlnet'])
        bert = BERT(self.config['bert'])
        xgb = XGBoost(self.config['xgboost'])
        self.model = VotingClassifier(estimators=[
            ('custom1', xlnet),
            ('custom2', bert),
            ('custom3', xgb)
        ], voting='hard')


    def train(self, X: np.array, y: np.array) -> int:
        y = np.array([-1 if y == 0 else 1 for y in y])
        self.model.fit(X, y)
        return 1
    
    def predict(self, X: np.array) -> np.array:
        return self.model.predict(X)
    
    def save(self, path: str) -> int:
        # save the model to a pickle file
        pickle.dump(self.model, open(path, 'wb'))
        return 1
    
    def load_model(self, path: str) -> int:
        # load the model from a pickle file
        self.model = pickle.load(open(path, 'rb'))
        return 1


