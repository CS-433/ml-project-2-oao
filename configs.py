from TextProc import preproc_configs
from Models import model_configs


bert_config_train = {
    'model_config': model_configs.bert_base_train,
    'preproc_config': preproc_configs.preproc_config_1
}

bert_config_train_2 = {
    'model_config': model_configs.bert_large_train,
    'preproc_config': preproc_configs.preproc_config_2
}