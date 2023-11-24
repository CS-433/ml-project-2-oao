from datetime import datetime
from Classifiers import Classifier
from TextProc import Preprocessor
from Models import model_configs
import argparse

class PredictEngine():
    def __init__(self, config, predict_data_path, output_path, verbose):
        self.config = config
        self.predict_data_path = predict_data_path
        self.output_path = output_path
        self.verbose = verbose

        # intialize classifier and preprocessor
        self.model = Classifier(config['model_config'])
        self.preprocessor = Preprocessor(config['preproc_config'])


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
    parser.add_argument('--config', type=str, default='bert_base')
    parser.add_argument('--metrics', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--predict_data_path', type=str, default='Data/test.csv')
    parser.add_argument('--output_path', type=str, default='Results/predictions_{}.csv'.format(current_time))
    parser.add_argument('--metrics_path', type=str, default='Results/metrics_{}.csv'.format(current_time))
    args = parser.parse_args()

    # load configuration
    config = model_configs.load_config(args.config)

    # initialize engine
    engine = PredictEngine(config, args.predict_data_path, args.output_path, args.verbose)

    # run engine
    engine.run()
