

bert_base_train = {
    'model_type': 'bert',
    'model_name': 'bert-base-uncased',
    'max_seq_length': 128,
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_proportion': 0.1,
    'num_labels': 2,
    'seed': 42
}

bert_large_train = {
    'model_type': 'bert',
    'model_name': 'bert-large-uncased',
    'max_seq_length': 128,
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_proportion': 0.1,
    'num_labels': 2,
    'seed': 42
}