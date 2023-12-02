from Classifiers.Classifier import Classifier

from simpletransformers.classification import ClassificationModel
import numpy as np
import pandas as pd
import pickle



class XLNet(Classifier):
    def __init__(self, config: dict):
        config = config.copy()
        config.pop('model_type')
        config.pop('model_name')

        super().__init__(config)
        if self.model == None:
            self.build_xlnet_classifier()
        

    def load_model(self, path: str) -> int:
        # load the model from a pickle file
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return 1

    def train(self, X: np.array, y: np.array) -> int:
        # concatenate X and y into a dataframe
        train_df = pd.DataFrame()
        train_df["text"] = X
        train_df["labels"] = y
        # train the model
        self.model.train_model(train_df)
        return 1

    def predict(self, X: np.array) -> np.array:
        X = X.tolist()
        preds, raw_out = self.model.predict(X)
        preds = np.array([-1 if pred == 0 else 1 for pred in preds])
        return preds

    def save(self, path: str) -> int:
        # save the model to a pickle file
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
    
    def build_xlnet_classifier(self):
        """
        Builds the XLNet classifier
        :return: None
        """
        self.model = ClassificationModel(
            "xlnet",
            "xlnet-base-cased",
            num_labels=2,
            use_cuda=False,
            args=self.config,
        )