from helper import *
from preproc_configs import config_1

class Preprocessor():
    def __init__(self, process_config):
        self.process_config = process_config

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
    

if __name__ == '__main__':
    # Example 1
    # Description: Preprocess a string using the preprocessing configuration config_1
    # Expected Result: 'preprocess string preprocessing configuration config'
    print(Preprocessor(config_1).preprocess_string('Preprocess a string using the preprocessing configuration config_1'))
    
    # Example 2
    # Description: Here we can see the preprocessing configuration config_1
    print(Preprocessor(config_1).preprocess_string('Here we can see the preprocessing configuration config_1'))