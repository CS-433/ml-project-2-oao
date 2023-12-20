bert_base_train = {
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

bert_large_train = {
    'model_type': 'bert',
    'model_encoder': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'model_preproc': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'max_seq_length': 200,
    'batch_size': 64,
    'epochs': 3,
    'lr': 4e-5,
    'warmup': 0.06,
    'validation_split': 0.15,
}

xlnet_base_train = {
    'model_type': 'xlnet',
    'model_name': 'xlnet-base-cased',
    'overwrite_output_dir': True,
    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,
    'train_batch_size': 64,
   'max_seq_length': 200,
   'validation_split': 0.15,
   'num_train_epochs': 10,
   'learning_rate': 4e-5,
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

bert_base_part_test = {
    'model_type': 'bert',
    'model_path': 'Models/BERT/AlisBERTPart/'
}

bert_base_full_test = {
    'model_type': 'bert',
    'model_path': 'Models/BERT/AlisBERTFull/'
}

xlnet_base_part_test = {
    'model_type': 'xlnet',
    'model_path': 'Models/XLNet/AlisXLNetPart/best_model/'
}

xlnet_base_full_test = {
    'model_type': 'xlnet',
    'model_path': 'Models/XLNet/AlisXLNetFull/best_model/'
}

xgboost_base_part_test = {
    'model_type': 'xgboost',
    'xgb_config': {
        'model_path': 'Models/XGBoost/AlisXGBPart'
    },
    'vectorizer_config': {
        'vectorizer': "word2vec"
    }
}

xgboost_full_test = {
    'model_type': 'xgboost',
    'xgb_config': {
        'model_path': 'Models/XGBoost/AlisXGBFull'
    },
    'vectorizer_config': {
        'vectorizer': "word2vec"
    }
}

ensemble_train = {
    'model_type': 'ensemble',
    'xlnet': xlnet_base_full_test,
    'bert': bert_base_full_test,
    'xgboost': xgboost_base_part_test
}