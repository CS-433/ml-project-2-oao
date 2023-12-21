[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13012101&assignment_repo_type=AssignmentRepo)

Welcome to the OAO team project! This project is a text classification project done in the context of our CS-433 Machine Learning course at EPFL.\
The goal of this project is to classify tweets as being either containing a smiling face or a sad face.\
More concretely, we aim to predict the sentiment of tweets by classifying them as being either positive or negative.

To do so, we have trained 3 different models:
- `BERT`: a transformer-based model
- `(RoBERTa)`: a transformer-based model
- `XLNet`: a transformer-based model
- `XGBoost`: a gradient boosting model

We have also implemented a text preprocessing pipeline that can be used to preprocess the data before training the models, as we know that social media language is often noisy and contains a lot of typos, abbreviations, etc.

## Project structure
The project is structured as follows:
- `Data/`: contains the data used for training and testing the model
- `Models/`: contains the saved models used for inference
- `Notebooks/`: contains some notebooks used for data exploration and model training
- `Classifiers/`: contains the code for different classifier classes used for training and inference
- `Results/`: contains the predictions made by the models on the test set
- `TextProc/`: contains the code for our text preprocessing methods
- `run.py`: is the main script used for training the models and inference on the test set

***

## Quick start
We have implemented 4 different Classifiers that can be used for inference, if you just want to quickly check the results of the models, please follow the instructions below.
1. Clone the repository
2. Download the `Models/` folder from the following link and place it at the root of the project [here](https://drive.google.com/drive/folders/1YxYE67JtN8F41HPGZvOdhdglz1nK1U87?usp=sharing)\
**Please Make Sure that the Models folders is at the root of the project and do not change the name of the folders inside as the scripts depends on that specific structure**
4. If you don't have the requirements installed, the cells in notebook QuickStart.ipynb will install them for you.
3. Then you simply used the following command to make predictions on the test set:
```bash
python run.py --config {test_config_name} --save_result {predictions_name}
```
where `test_config_name` is one of the following:
- `bert_config_full_test`
- `xlnet_config_part_test`
- `xgboost_config_full_test`
- `roberta_config_test`

And `predictions_name` is the name of the file where to save the predictions on the test set. You will find the predictions under `Results/{predictions_name}.csv`.

***

## Train your own model
You can run the code by executing the `run.py` script.\
The script takes the following arguments:
- `--config`: the configuration file for training the model or predicting
- `--train`: whether to train the model or not (default: `False`)
- `train_data_path`: the path to the training data (default: `Data/twitter-datasets/`)
- `--save_model_path`: if training a new model, the path where to save the model (default: `Models/{model_type}`)
- `--save_result`: the path where to save the predictions on the test set (default: `predictions`)

If you want to train one of the models, use the following command:
```bash
python run.py --train --config {config_name} --save_model_path {path_to_save_model}
```
The `config_name` is the name of the configuration file for training the model and can be found in the `configs.py` file.\
The `path_to_save_model` is the path where to save the model, the saved model will be found under `Models/{model_type}/{path_to_save_model}`.

If you want to make predictions on the test set, use the following command:
```bash
python run.py --config {test_config_name} --save_result {predictions_name}
```
The `test_config_name` is the name of the configuration for predicting and can be found in the `configs.py` file.\
Note that if you have trained a model and saved it, you need to create a `test_model_config` in the `model_configs.py` file.\
as well as a general configuration in the `configs.py` file.\
More details below.

Here are some example commands:
```bash
python run.py --config bert_base_part_train --save_result bert_base_part_predictions
```
```bash
python run.py --config xlnet_base_part_train --save_result xlnet_base_part_predictions
```
```bash
python run.py --config xgboost_base_part_train --save_result xgboost_base_part_predictions
```

***
## Configuration Structure
This project is based on a configuration system that dictates for instance, parameters used for training the model, but also the type of model to use, the type of preprocessing to apply to the data, etc.\
There are 3 configuration files:
- `configs.py`: contains the general configuration for the project
- `model_configs.py`: contains the configuration for the model
- `preproc_configs.py`: contains the configuration for the preprocessing

### Model configuration
Let's take a look at the structure of the `model_configs.py` file:
#### Training configuration
Since we use different models, the configuration file for each model type will look slightly different.\
For instance, the configuration file for training `BERT` model looks like this:
```python
    bert_train = {
        'model_type': 'bert',
        'model_encoder': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
        'model_preproc': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'max_seq_length': 200,
        'batch_size': 128,
        'epochs': 5,
        'warmup': 0.1,
        'lr': 6e-10,
        'validation_split': 0.15,
    }
```
As you can see, the configuration file contains the following parameters:
- `model_type`: the type of model to use
- `model_encoder`: the encoder to use for the model
- `model_preproc`: the preprocessing to apply to the data
- `max_seq_length`: the maximum length of the input sequence
- `batch_size`: the batch size to use for training
- `epochs`: the number of epochs to train the model
- `warmup`: the warmup proportion for the learning rate scheduler
- `lr`: the learning rate to use for training
- `validation_split`: the proportion of the training data to use for validation

The structure of the configuration file for training `XLNet`and `XGBoost` looks like this:
```python
    xlnet_train = {
        'model_type': 'xlnet',
        'max_seq_length': 200,
        'batch_size': 128,
        'epochs': 5,
        'warmup': 0.1,
        'lr': 6e-10,
        'validation_split': 0.15,
    }
```
```python
    xgboost_base_train = {
        'model_type': 'xgboost',
        'xgb_config': {
            'learning_rate': 0.189,
            'max_depth': 10,
            'n_estimators': 742,
            'subsample': 0.881,
        },
        'vectorizer_config': {
            'vectorizer': "word2vec",
            'vector_size': 350,
            'window': 5,
            'min_count': 1,
            'workers': 4,
            'sg': 1
        }
    }
```
In the case of `XGBoost`, if you specify the `vectorizer_config` parameter, the model will be trained using the `Word2Vec` vectorizer.\
Which also needs some specific parameters to be specified.

#### Inference configuration
The inference configuration looks the same accross all models, we only need to specify the model type and the path to the saved model.\
For instance, the inference configuration for `BERT` looks like this:
```python
    bert_base_part_test = {
        'model_type': 'bert',
        'model_path': 'Models/BERT/AlisBERTPart/'
    }
```

### Preprocessing configuration
Now let's take a look at the structure of the `preproc_configs.py` file:\
A preprocessing configuration is simply a list of preprocessing functions to apply to the data.\
We have curated two configurations for our specific use cases, but you can simply create your own configuration by adding the preprocessing functions you want to apply to the data.\
All functions are available in the `TextProc/helper.py` file.
Here is an example configuration:
```python
    preproc_config = [
        remove_punctuation,
        remove_stopwords,
        to_lowercase,
        remove_numbers,
        remove_whitespace
    ]
```

### General configuration
Finally, let's take a look at the structure of the `configs.py` file:\
A general configuration is simply a mix of the model configuration and the preprocessing configuration.\
For instance, the configuration for training `BERT` looks like this:
```python
    bert_base_part_train = {
        'model_config': model_configs.bert_base_train,
        'preproc_config': preproc_configs.preproc_config_context
    }
```
**It is this configuration name `bert_base_part_train` that you will need to specify when running the `run.py` script.**\
And the same holds for the inference configuration.

***
## Classifiers Structure
Now let's dive into the structure of the `Classifiers` folder.\
Since our project aims at comparing different models (BERT, XLNet, XGBoost), we have created a class for each model type.\
All of the classes inherit from the `Classifier` abstract class, which contains the basic methods for training and inference.\
Hence all of the classes provide the same set of methods, but the implementation of these methods is different for each model type.\
Each classifier provides the following methods:
- `train`: trains the model
- `predict`: makes predictions on the test set
- `save`: saves the model
- `load_model`: loads the model
- `validate`: validates the model on the validation set

***
## Text preprocessing
The text preprocessing is done using the `TextProc` package.\
This package offers a `Preprocessor` class that takes a list of preprocessing functions as input and applies them to the data.\
The `Preprocessor` can be set to **verbose** mode, which will print the data before and after each preprocessing step (useful for debugging).\
Details about the preprocessing functions can be found in our report.

***
## Estimated running time
| Model | Epochs | Time per epoch |
|---------|-----------|--------|
| Bert | 6 | 30min |
| XLNet | 6 | 30min |
| XGBoost | -- | 10 min |





