from Classifiers.Classifier import Classifier

print('Loading tensorflow...')
import tensorflow as tf
import tensorflow_hub as hub
print('Finished loading tensorflow')
import numpy as np

print('Loading transformers...')
from transformers import TFBertModel, BertTokenizer, DistilBertTokenizer, RobertaTokenizer
print('Finished loading transformers')


import pandas as pd

tf.get_logger().setLevel('ERROR')


class BERT(Classifier):
    def __init__(self, config: dict):
        super().__init__(config)
        if self.model == None:
            self.build_bert_classifier(config['model_name'])
        

    
    def train(self, X: list, y: list) -> None:
        """
        trains the model
        :param data: training data
        """
        # encode the data
        print('Encoding data...')
        X_ids, X_bert_encoded = self.bert_encode(X)
        # train the model
        print('Training model...')
        self.model.fit(
            [X_ids, X_bert_encoded],
            y,
            validation_split = self.config['validation_split'],
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )


    def bert_encode(self, texts: np.array) -> list:
        """
        encodes the data using bert tokenizer
        :param data: data to be encoded
        :return: encoded data
        """
        encoded_dict = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.config['max_seq_length'],
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )
        return np.array(encoded_dict['input_ids']), np.array(encoded_dict['attention_mask'])

    def build_bert_classifier(self, bert_model_name):
        self.model = TFBertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        input_ids = tf.keras.Input(shape=(60,),dtype='int32')
        attention_masks = tf.keras.Input(shape=(60,),dtype='int32')
        
        output = self.model([input_ids,attention_masks])
        output = output[1]
        
        output = tf.keras.layers.Dense(32,activation='relu')(output)
        output = tf.keras.layers.Dropout(0.2)(output)

        output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
        self.model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
        self.model.compile(tf.keras.optimizers.Adam(learning_rate=self.config['lr']), loss=self.config['loss'], metrics=['accuracy'])

    def predict(self, X) -> list:
        """
        predicts the labels for the data
        :param data: data to be predicted
        :return: predictions
        """
        X_ids, X_bert_encoded = self.bert_encode(X)
        return self.model.predict([X_ids, X_bert_encoded])

    def save(self, path: str) -> None:
        """
        saves the model
        :param path: path to save the model
        """
        # save the bert model to the given path
        self.model.save(path, include_optimizer=False)

    def load_model(self, path: str) -> None:
        """
        loads the model
        :param path: path to load the model
        """
        self.model = tf.keras.models.load_model(path)