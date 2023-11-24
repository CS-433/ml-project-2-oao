from datetime import datetime
from Classifiers import Classifier
from TextProc import Preprocessor

import argparse
import pandas as pd
import numpy as np

class TrainEngine():
    def __init__(self, config, train_data_path, verbose):
        self.config = config
        self.train_data_path = train_data_path
        self.verbose = verbose

        # intialize classifier and preprocessor
        self.model = Classifier(config['model_config'])
        self.preprocessor = Preprocessor(config['preproc_config'])


    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        loads data from a csv file
        :param path: path to csv file
        :return: pandas dataframe
        """
        return pd.read_csv(data_path)
    
    def save_metrics(self, metrics: dict, path: str) -> None:
        """
        saves metrics to a csv file
        :param metrics: metrics to be saved
        :param path: path to csv file
        """
        pd.DataFrame.from_dict(metrics, orient='index').to_csv(path)

    def save_predictions(self, predictions: list, path: str) -> None:
        """
        saves predictions to a csv file
        :param predictions: predictions to be saved
        :param path: path to csv file
        """
        pd.DataFrame(predictions).to_csv(path)


    def run(self):
        print('Loading data...')
        # load data
        train_data = self.load_data(self.train_data_path)
        print('Preprocessing data...')
        # preprocess data
        train_data = self.preprocessor.preprocess(train_data)
        print('Training...')
        # train
        self.model.train(train_data)
        print('Saving model...')
        # save model
        self.model.save(self.model_path)
        print('Testing...')
        # test
        test_data = self.load_data(self.test_data_path)
        test_data = self.preprocessor.preprocess(test_data)
        predictions = self.model.predict(test_data)
        print('Saving predictions...')
        # save predictions
        self.save_predictions(predictions, self.output_path)
        print('Saving metrics...')
        # save metrics
        self.save_metrics(self.model.get_metrics(), self.metrics_path)
        print('Done.')


if __name__ == "main":
    """
    This file is used to train a model on a given training dataset
    using the specified configurations.
    """
    # getting current date time for saving results and metrics
    now = datetime.now()
    current_time = now.strftime("%m-%d-%Y-%H:%M:%S")

    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='bert_base')
    parser.add_argument('--train_data_path', type=str, default='Data/train.csv')
    parser.add_argument('test_data_path', type=str, default='Data/test.csv')
    parser.parse_args('model_path', type=str, default='Models/model_{}.pt'.format(current_time))

    args = parser.parse_args()