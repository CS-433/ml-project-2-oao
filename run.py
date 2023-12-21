print('Loading Classifiers...')
from Classifiers.Classifier import Classifier
from Classifiers.BERT import BERT
from Classifiers.XGBoost import XGBoost
from Classifiers.XLNet import XLNet
from Classifiers.RoBERTa import RoBERTa
print('Finished loading Classifiers')

from TextProc.Preprocessor import Preprocessor

import configs

import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
# import train test validation split from sklearn
from sklearn.model_selection import train_test_split


# print current directory
import os
print(os.getcwd())

class TrainEngine():
    def __init__(self, train, config, train_data_path, save_path, save_result_path):
        self.train = train
        self.config = config
        self.train_data_path = train_data_path
        self.save_model_path = save_path
        self.save_result_path = 'Results/' + save_result_path + '.csv'

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
        pos_path = '{}/train_pos/partial_context.txt'.format(data_path)
        neg_path = '{}/train_neg/partial_context.txt'.format(data_path)
        test_path = '{}/test/test_context.txt'.format(data_path)

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

    def save_predictions(self, predictions: pd.DataFrame, path: str) -> None:
        """
        saves predictions to a csv file
        :param predictions: predictions to be saved
        :param path: path to csv file
        """
        predictions.to_csv(path, index=False)

    def choose_model(self, model_config: dict) -> Classifier:
        """
        chooses a model based on the model_type
        :param model_type: type of model to be chosen
        :return: model
        """
        if model_config['model_type'] == 'bert':
            self.save_model_path = 'Models/BERT/' + self.save_model_path
            return BERT(model_config)
        elif model_config['model_type'] == 'xgboost':
            self.save_model_path = 'Models/XGBoost/' + self.save_model_path
            return XGBoost(model_config)
        elif model_config['model_type'] == 'xlnet':
            output_dir = 'Models/XLNet/' + self.save_model_path + '/output/'
            tensorboard_dir = 'Models/XLNet/' + self.save_model_path + '/tensorboard_dir/'
            best_model_dir = 'Models/XLNet/' + self.save_model_path + '/best_model/'
            mc = model_config.copy()
            mc['output_dir'] = output_dir
            mc['best_model_dir'] = best_model_dir
            mc['tensorboard_dir'] = tensorboard_dir
            return XLNet(mc)
        elif model_config['model_type'] == 'roberta':
            self.save_model_path = 'Models/RoBERTa/' + self.save_model_path
            return RoBERTa(model_config)
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

            print('Preprocessing data...')
            # preprocess data
            # X = self.preprocessor.preprocess(X)

            # drop label column
            X.drop('label', axis=1, inplace=True)

            # split data into train and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.005)
            # convert to list
            X_train = np.array(X_train['text'].tolist())
            X_val = np.array(X_val['text'].tolist())
            y_train = np.array(y_train.tolist())
            y_val = np.array(y_val.tolist())
            y_val = np.array([-1 if y == 0 else 1 for y in y_val])


            print('Training...')
            # train and print progression using tqdm
            tqdm(self.model.train(X_train, y_train))


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
            # X = self.preprocessor.preprocess(X)
            # convert to list
            X_val = X['text'].to_numpy()

            print('Predicting...')
            # predict
            X['Prediction'] = self.model.predict(X_val)

            print('Saving predictions...')
            # save predictions without the index
            self.save_predictions(X[['Id', 'Prediction']], self.save_result_path)

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
    parser.add_argument('--config', type=str, default='bert_config_train', required=False)
    # add a boolean argument to specify whether to train or test
    parser.add_argument('--train', action='store_true', required=False)
    parser.add_argument('--train_data_path', type=str, default='Data/twitter-datasets', required=False)
    parser.add_argument('--save_model_path', type=str, default='model_{}'.format(current_time), required=False)
    parser.add_argument('--save_result', type=str, default='Results/predictions.csv', required=False)

    args = parser.parse_args()

    print('Loading config...')
    # loading config
    config = configs.get_config(args.config)

    print('Initializing train engine...')
    # initializing train engine
    print('this is train:', args.train)
    engine = TrainEngine(args.train, config, args.train_data_path, args.save_model_path, args.save_result)

    print('Running training...')
    # running training
    engine.run()
