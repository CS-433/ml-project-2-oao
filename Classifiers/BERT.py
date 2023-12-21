from Classifiers.Classifier import Classifier

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from official.nlp import optimization
from tqdm import tqdm
tf.get_logger().setLevel('ERROR')


class BERT(Classifier):
    def __init__(self, config: dict):
        super().__init__(config)
        if self.model == None:
            self.build_bert_classifier()
        
    def load_model(self, path: str) -> int:
        """
        loads the model
        :param path: path to load the model
        """
        self.model = tf.saved_model.load(path)
        return 1
    
    def train(self, X: np.array, y: np.array) -> int:
        """
        Trains the Bert Classifier
        :param X: training data
        :param y: training labels
        :return: history of the training

        Make sure to specify the following in the config:
        - batch_size: batch size
        - epochs: number of epochs
        - warmup: warmup ratio
        - lr: learning rate
        - validation_split: validation split ratio
        """
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()

        steps_per_epoch = len(X) // self.config['batch_size']
        num_train_steps = steps_per_epoch * self.config['epochs']
        num_warmup_steps = int(self.config['warmup']*num_train_steps)
        optimizer = optimization.create_optimizer(init_lr=self.config['lr'],
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')
        
        self.model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
        
        history = self.model.fit(x=X,
                                 y=y,
                                 validation_split=self.config['validation_split'],
                               epochs=self.config['epochs'])
        
        return 1

    def build_bert_classifier(self) -> int:
        """
        Builds the bert model
        The model is built using the tensorflow hub

        Make sure to specify the following in the config:
        - bert_preproc: path to the preprocessing layer
        - bert_encoder: path to the bert encoder

        :return: True if the model was built successfully
        """
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.config['model_preproc'], name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.config['model_encoder'], trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        self.model = tf.keras.Model(text_input, net)
        return 1

    def predict(self, X: np.array) -> np.array:
        """
        predicts the labels for the data
        :param data: data to be predicted
        :return: predictions
        """
        bar = tqdm(total=len(X), desc='Predicting')

        batch_size = 100
        predictions = []
        for i in range(0, len(X), batch_size):
            batch_examples = X[i:i+batch_size]
            preds = tf.sigmoid(self.model(tf.constant(batch_examples)))
            predictions.extend(preds.numpy().flatten().tolist())

            # Update the progress bar
            bar.update(len(batch_examples))

        # Close the progress bar
        bar.close()
        predictions = np.array([1 if p > 0.5 else -1 for p in predictions])
        return predictions

    def save(self, path: str) -> int:
        """
        saves the model
        :param path: path to save the model
        """
        self.model.save(path, include_optimizer=False)
        return 1