

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
    'train_batch_size': 32,
   'max_seq_length': 64,
   'num_train_epochs': 4,
   'learning_rate': 4e-5,
   'adam_epsilon': 1e-6,
   'warmup_ratio': 0.06,
   'warmup_steps': 1,
   'eval_all_checkpoints': True,
   'reprocess_input_data': False,
}

bert_base_test = {
    'model_type': 'bert',
    'model_path': 'Models/model_12-02-2023-19:51:52'
}