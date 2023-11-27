from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Classifier(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        if 'model_path' in config:
            self.load_model(config['model_path'])
            
        self.metrics = {}

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    
    def get_metrics(self):
        return self.metrics.copy()
    
    
    def validate(self, X, y):
        """
        validates the model
        :param X: validation data
        :param y: validation labels
        :return: metrics
        """
        # predict labels
        y_pred = self.predict(X)
        # calculate metrics for nlp models
        self.metrics['accuracy'] = accuracy_score(y, y_pred)
        self.metrics['precision'] = precision_score(y, y_pred, average='macro')
        self.metrics['recall'] = recall_score(y, y_pred, average='macro')
        self.metrics['f1'] = f1_score(y, y_pred, average='macro')