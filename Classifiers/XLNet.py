from Classifiers.Classifier import Classifier

from simpletransformers.classification import ClassificationModel
import numpy as np
import pandas as pd
import pickle
import torch



class XLNet(Classifier):
    def __init__(self, config: dict):
        config = config.copy()
        if 'model_type' in config:
            config.pop('model_type')
        if 'model_name' in config:
            config.pop('model_name')

        super().__init__(config)
        if self.model == None:
            self.build_xlnet_classifier()
        

    def load_model(self, path: str) -> int:
        # load the model from a pickle file
        cuda_available = torch.cuda.is_available()
        self.model = ClassificationModel(
            "xlnet",
            path,
            use_cuda=torch.cuda.is_available(),
        )
        return 1

    def train(self, X: np.array, y: np.array) -> int:
        # concatenate X and y into a dataframe
        train_df = pd.DataFrame()
        train_df["text"] = X
        train_df["labels"] = y
        # split the data into training and validation sets
        val_df = train_df.sample(frac=self.config['validation_split'])
        train_df.drop(val_df.index, inplace=True)

        # train the model
        self.model.train_model(train_df, eval_df=val_df)
        return 1

    def predict(self, X: np.array) -> np.array:
        X = X.tolist()
        preds, raw_out = self.model.predict(X)
        preds = np.array([-1 if pred == 0 else 1 for pred in preds])
        return preds

    def save(self, path: str) -> int:
        pass
    
    def build_xlnet_classifier(self):
        """
        Builds the XLNet classifier
        :return: None
        """
        cuda_available = torch.cuda.is_available()
        self.model = ClassificationModel(
            "xlnet",
            "xlnet-base-cased",
            num_labels=2,
            use_cuda=cuda_available,
            args=self.config,
        )