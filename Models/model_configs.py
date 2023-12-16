from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec

bert_base_train = {
    'model_type': 'bert',
    'model_encoder': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'model_preproc': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'max_seq_length': 128,
    'batch_size': 32,
    'epochs': 5,
    'warmup': 0.1,
    'lr': 6e-6,
    'validation_split': 0.15
}

bert_large_train = {
    'model_type': 'bert',
    'model_name': 'small_bert/bert_en_uncased_L-4_H-512_A-8',
    'max_seq_length': 128,
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_proportion': 0.1,
    'num_labels': 2,
    'seed': 42
}

xlnet_base_train = {
    'model_type': 'xlnet',
    'model_name': 'xlnet-base-cased',
    'overwrite_output_dir': True,
    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,
    'train_batch_size': 512,
   'max_seq_length': 200,
   'validation_split': 0.15,
   'num_train_epochs': 10,
   'learning_rate': 4e-10,
   'adam_epsilon': 1e-6,
   'warmup_ratio': 0.06,
   'warmup_steps': 1,
   'no_cache': True,
   'evaluate_during_training': True,
   'evaluate_during_training_verbose': True,
   'reprocess_input_data': False,
}

xgboost_base_train = {
    'model_type': 'xgboost',
    'xgb_config': {
        'learning_rate': 0.189,
        'max_depth': 10,
        'n_estimators': 800,
        'subsample': 0.881,
    },
    'vectorizer_config': {
        'vectorizer': "word2vec",
        'vector_size': 300,
        'window': 5,
        'min_count': 1,
        'workers': 4,
    }
}

bert_base_test = {
    'model_type': 'bert',
    'model_path': 'Models/model_12-02-2023-19:51:52'
}

xlnet_base_test = {
    'model_type': 'xlnet',
    'model_path': 'Models/XLNet/model_12-06-2023-10:47:04/best_model/'
}

xgboost_base_test = {
    'model_type': 'xgboost',
    'xgb_config': {
        'model_path': 'Models/XGBoost/AlisXGB'
    },
    'vectorizer_config': {
        'vectorizer': "word2vec"
    }
}