from Classifiers.Classifier import Classifier
import numpy as np
import xgboost as xgb
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim


class XGBoost(Classifier):
    def __init__(self, config: dict):
        super().__init__(config['xgb_config'])
        self.config = config

    def load_model(self, path: str) -> int:
        if os.path.exists(path):
            self.model = pickle.load(open(path+'/model.pkl', 'rb'))
            # load vectorizer
            # if therer exists a file named vectorizer in the path, load the vectorizer
            if os.path.exists(path+'/vectorizer.pkl'):
                self.vectorizer = pickle.load(open(path+'/vectorizer.pkl', 'rb'))
            return 0  # Model loaded successfully
        else:
            return -1  # Model file not found

    def train(self, X: np.array, y: np.array):
        self.vectorizer = XGBoost.Vectorizer(self.config['vectorizer_config'], X)
        X = self.vectorizer.vectorize(X)
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.config['xgb_config'], dtrain)


    def predict(self, X: np.array) -> np.array:
        X = self.vectorizer.vectorize(X)
        dtest = xgb.DMatrix(X)
        predictions = [1 if p >= 0.5 else -1 for p in self.model.predict(dtest)]
        return np.array(predictions)

    def save(self, path: str) -> int:
        if self.model:
            pickle.dump(self.model, open(path+'/model.pkl', 'wb'))
            # save vectorizer
            print('this is the vectorizer type', self.config['vectorizer_config']['vectorizer'])
            if self.config['vectorizer_config']['vectorizer'] == 'word2vec':
                pickle.dump(self.vectorizer, open(path+'/vectorizer.pkl', 'wb'))
            return 0  # Model saved successfully
        else:
            return -1  # No model to save
        

    class Word2Vec():
        def __init__(self, config: dict, texts: list):
            texts = [text.split() for text in texts]
            if 'vector_size' in config:
                self.wv = gensim.models.Word2Vec(sentences=texts, vector_size= config['vector_size'], window=config['window'], min_count=config['min_count'], workers=config['workers'])
            else:
                self.wv = None

        def fit_transform(self, X: list) -> list:
            # vectorize each word in the sentence and take the mean
            X = [text.split() for text in X]
            vectorized_texts = []
            for words in X:
                vectors = [self.wv.wv[word] for word in words if word in self.wv.wv]
                if vectors:
                    vectors = np.mean(vectors, axis=0)
                else:
                    vectors = np.zeros(self.wv.vector_size)
                vectorized_texts.append(vectors)
            return np.array(vectorized_texts)
        

    class Vectorizer():
        def __init__(self, config: dict, texts: list):
            self.config = config
            self.vectorizer = config['vectorizer']
            if self.vectorizer == 'tfidf':
                self.vectorizer = TfidfVectorizer()
            elif self.vectorizer == 'count':
                self.vectorizer = CountVectorizer()
            elif self.vectorizer == 'word2vec':
                self.vectorizer = XGBoost.Word2Vec(config, texts=texts)

        def vectorize(self, X: np.array) -> np.array:
            return self.vectorizer.fit_transform(X)
        