from TextProc import preproc_configs
from Models import model_configs


configurations = {
    'bert_config_train' : {
        'model_config': model_configs.bert_base_train,
        'preproc_config': preproc_configs.preproc_config_1
    },

    'bert_config_train_2' : {
        'model_config': model_configs.bert_large_train,
        'preproc_config': preproc_configs.preproc_config_2
    },

    'xlnet_config_train' : {
        'model_config': model_configs.xlnet_base_train,
        'preproc_config': preproc_configs.preproc_config_1
    },

    'bert_config_test' : {
        'model_config': model_configs.bert_base_test,
        'preproc_config': preproc_configs.preproc_config_1
    },
}

def get_config(config_name: str):
    """
    returns the configuration with the given name
    :param config_name: name of the configuration
    :return: configuration
    """
    return configurations[config_name]