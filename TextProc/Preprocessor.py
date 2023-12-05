from TextProc.helper import *
from TextProc.preproc_configs import preproc_config_1
from tqdm import tqdm

import pandas as pd

class Preprocessor():
    def __init__(self, process_config, verbose: bool = False):
        self.process_config = process_config
        self.verbose = verbose

    def preprocess_string(self, text: str) -> str:
        """
        preprocesses a text by applying all functions
        specified in the preprocessing configuration
        :param text: text to be preprocessed
        :return: preprocessed text
        """
        for func in self.process_config:
            text = func(text)
        return text
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        preprocesses a dataframe by applying all functions
        specified in the preprocessing configuration
        :param data: data to be preprocessed
        :return: preprocessed data
        """
        texts = data['text'].tolist()
        for func in self.process_config:
            # print the name of the function
            print(func.__name__)
            # use tqdm to show progress bar gradually
            for i in tqdm(range(len(texts))):
                texts[i] = func(texts[i])
            if self.verbose:
                print(data.head())

        data['text'] = texts
        return data