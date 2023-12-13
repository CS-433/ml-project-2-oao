from TextProc.helper import *
from TextProc.preproc_configs import preproc_config_1
from tqdm import tqdm

import pandas as pd

tqdm.pandas()

class Preprocessor():
    def __init__(self, process_config, verbose: bool = False):
        self.process_config = process_config
        self.verbose = verbose

    def preprocess_string(self, texts: list) -> list:
        """
        preprocesses a text by applying all functions
        specified in the preprocessing configuration
        :param text: text to be preprocessed
        :return: preprocessed text
        """
        for func in self.process_config:
            texts = [func(texts[0])]
            if self.verbose:
                print(texts[0])
        return texts
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        preprocesses a dataframe by applying all functions
        specified in the preprocessing configuration
        :param data: data to be preprocessed
        :return: preprocessed data
        """
        # create a copy of the dataframe
        data = data.copy()
        for func in self.process_config:
            # print the name of the function
            print(func.__name__)
            if func.__name__ == 'apply_spacy':
                data['text'] = apply_spacy(data['text'].tolist())
            else:
                # apply the function to the data and print the progression
                data['text'] = data['text'].progress_apply(func)

            if self.verbose:
                print(data['text'].iloc[0])

        return data