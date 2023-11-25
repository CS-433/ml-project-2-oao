

bert_base_train = {
    'model_type': 'bert',
    'model_name': 'bert-base-uncased',
    'max_seq_length': 128,
    'batch_size': 32,
    'epochs': 5,
    'validation_split': 0.1,
    'loss': 'binary_crossentropy',
    'lr': 6e-6,
    'validation_split': 0.15
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