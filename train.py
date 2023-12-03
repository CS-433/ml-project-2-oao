print('Loading Classifiers...')
from Classifiers.Classifier import Classifier
from Classifiers.BERT import BERT
from Classifiers.XGBoost import XGBoost
from Classifiers.XLNet import XLNet
print('Finished loading Classifiers')

from TextProc.Preprocessor import Preprocessor

import configs

import argparse
from datetime import datetime
import pandas as pd
import numpy as np
# import train test validation split from sklearn
from sklearn.model_selection import train_test_split


# print current directory
import os
print(os.getcwd())

class TrainEngine():
    def __init__(self, train, config, train_data_path, save_path):
        self.train = train
        self.config = config
        self.train_data_path = train_data_path
        self.save_model_path = save_path

        # intialize classifier and preprocessor
        self.model = self.choose_model(config['model_config'])
        self.preprocessor = Preprocessor(config['preproc_config'])


    def load_data(self, data_path: str, train: bool = True) -> pd.DataFrame:
        """
        loads data from a csv file
        :param path: path to csv file
        :return: pandas dataframe
        """
        # load the dataset of texts one text per line
        pos_path = '{}/train_pos.txt'.format(data_path)
        neg_path = '{}/train_neg.txt'.format(data_path)
        test_path = '{}/test_data.txt'.format(data_path)

        if train:
            with open(pos_path, 'r') as f:
                train_pos = f.readlines()

            with open(neg_path.format(data_path), 'r') as f:
                train_neg = f.readlines()

            # create a dataframe with the text and label
            df = pd.DataFrame({'text': np.concatenate([train_pos, train_neg]),
                                'label': np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
                                })
        else:
            with open(test_path, 'r') as f:
                test = f.readlines()
            df = pd.DataFrame({'text': test})
            # split text on the first comma
            df['Id'] = df['text'].apply(lambda x: x.split(',', 1)[0])
            df['text'] = df['text'].apply(lambda x: x.split(',', 1)[1])
        
        print('size of training data:', df.shape)
        return df

    
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

    def choose_model(self, model_config: dict) -> Classifier:
        """
        chooses a model based on the model_type
        :param model_type: type of model to be chosen
        :return: model
        """
        if model_config['model_type'] == 'bert':
            return BERT(model_config)
        elif model_config['model_type'] == 'xgboost':
            # add item to model_config
            model_config['output_dir'] = self.save_model_path
            model_config['best_model_dir'] = self.save_model_path + '/best_model/'
            return XGBoost(model_config)
        elif model_config['model_type'] == 'xlnet':
            return XLNet(model_config)
        else:
            raise ValueError('Model type not supported')

    def print_metrics(self, metrics: dict) -> None:
        """
        prints the metrics
        :param metrics: metrics to be printed
        """
        for key, value in metrics.items():
            print('{}: {}'.format(key, value))

    def run(self):
        print('Loading data...')
        X = self.load_data(self.train_data_path, train=self.train)
        if self.train:
            # load data
            y = X['label']
            # drop label column
            X.drop('label', axis=1, inplace=True)

            print('Preprocessing data...')
            # preprocess data
            X = self.preprocessor.preprocess(X)
            # split data into train and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            # convert to list
            X_train = np.array(X_train['text'].tolist())
            X_val = np.array(X_val['text'].tolist())
            y_train = np.array(y_train.tolist())
            y_val = np.array(y_val.tolist())
            y_val = np.array([-1 if y == 0 else 1 for y in y_val])

            print('Training...')
            # train
            self.model.train(X_train, y_train)

            print('Saving model...')
            # save model
            self.model.save(self.save_model_path)

            print('Testing...')
            # test
            self.model.validate(X_val, y_val)

            print('Metrics:')
            # get metrics
            metrics = self.model.get_metrics()
            self.print_metrics(metrics)
        else:
            # load data
            X = self.load_data(self.train_data_path, train=False)
            # preprocess data
            X['text'] = self.preprocessor.preprocess(X['text'].to_numpy())
            # convert to list
            X_val = X['text'].to_numpy()

            print('Predicting...')
            # predict
            X['label'] = self.model.predict(X_val)

            print('Saving predictions...')
            # save predictions
            self.save_predictions(X[['Id', 'Prediction']], self.save_model_path)

        print('Finished Running !')




        




if __name__ == '__main__':
    """
    This file is used to train a model on a given training dataset
    using the specified configurations.
    """
    # getting current date time for saving results and metrics
    print('Getting current date time...')
    now = datetime.now()
    current_time = now.strftime("%m-%d-%Y-%H:%M:%S")

    print('Parsing arguments...')
    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, default='bert_config_train', required=False)
    parser.add_argument('--train', type=bool, default=True, required=False)
    parser.add_argument('--train_data_path', type=str, default='Data/twitter-datasets', required=False)
    parser.add_argument('--save_model_path', type=str, default='Models/model_{}'.format(current_time), required=False)

    args = parser.parse_args()

    print('Loading config...')
    # loading config
    config = configs.get_config(args.train_config)

    print('Initializing train engine...')
    # initializing train engine
    engine = TrainEngine(args.train, config, args.train_data_path, args.save_model_path)

    print('Running training...')
    # running training
    engine.run()
