from TextProc.Preprocessor import Preprocessor

from Classifiers.Classifier import Classifier
from Classifiers.BERT import BERT
from Classifiers.XGBoost import XGBoost
from Classifiers.XLNet import XLNet

import configs

from datetime import datetime
import argparse
import pandas as pd

class PredictEngine():
    def __init__(self, config, predict_data_path, output_path, verbose):
        self.config = config
        self.predict_data_path = predict_data_path
        self.output_path = output_path
        self.verbose = verbose

        # intialize classifier and preprocessor
        self.model = self.choose_model(config['model_config'])
        self.preprocessor = Preprocessor(config['preproc_config'])

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        loads data from a csv file
        :param path: path to csv file
        :return: pandas dataframe
        """
        return pd.read_csv(data_path)
    
    def save_predictions(self, predictions: list, path: str) -> None:
        """
        saves predictions to a csv file
        :param predictions: predictions to be saved
        :param path: path to csv file
        """
        pd.DataFrame(predictions).to_csv(path)

    def choose_model(self, model_config: dict) -> Classifier:
        """
        chooses a model based on the model_type
        :param model_type: type of model to be chosen
        :return: model
        """
        if model_config['model_type'] == 'bert':
            return BERT(model_config)
        elif model_config['model_type'] == 'xgboost':
            return XGBoost(model_config)
        elif model_config['model_type'] == 'xlnet':
            return XLNet(model_config)
        else:
            raise ValueError('Invalid model type.')


    def run(self):
        if self.verbose:
            print('Loading data...')
        # load data
        data = self.load_data(self.predict_data_path)
        if self.verbose:
            print('Preprocessing data...')
        # preprocess data
        data = self.preprocessor.preprocess(data)
        if self.verbose:
            print('Predicting...')
        # predict
        predictions = self.model.predict(data)
        if self.verbose:
            print('Saving predictions...')
        # save predictions
        self.save_predictions(predictions, self.output_path)
        if self.verbose:
            print('Done.')

if __name__ == '__main__':
    # getting current date time for saving results and metrics
    now = datetime.now()
    current_time = now.strftime("%m-%d-%Y-%H:%M:%S")

    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_config', type=str, default='bert_base')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--predict_data_path', type=str, default='Data/test.csv')
    parser.add_argument('--output_path', type=str, default='Results/predictions_{}.csv'.format(current_time))
    args = parser.parse_args()

    # load configuration
    config = getattr(configs, args.test_config)

    # initialize engine
    engine = PredictEngine(config, args.predict_data_path, args.output_path, args.verbose)

    # run engine
    engine.run()
