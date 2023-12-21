from TextProc import preproc_configs
from Models import model_configs


configurations = {
    'bert_config_train' : {
        'model_config': model_configs.bert_base_train,
        'preproc_config': preproc_configs.preproc_config_context
    },

    'bert_config_train_2' : {
        'model_config': model_configs.bert_large_train,
        'preproc_config': preproc_configs.preproc_config_2
    },

    'xlnet_config_train' : {
        'model_config': model_configs.xlnet_base_train,
        'preproc_config': preproc_configs.preproc_config_context
    },

    'xlnet_config_part_test' : {
        'model_config': model_configs.xlnet_base_part_test,
        'preproc_config': preproc_configs.preproc_config_context
    },

    'xlnet_config_full_test' : {
        'model_config': model_configs.xlnet_base_full_test,
        'preproc_config': preproc_configs.preproc_config_context
    },

    'bert_config_part_test' : {
        'model_config': model_configs.bert_base_part_test,
        'preproc_config': preproc_configs.preproc_config_context
    },

    'bert_config_full_test' : {
        'model_config': model_configs.bert_base_full_test,
        'preproc_config': preproc_configs.preproc_config_context
    },

    'xgboost_config_train' : {
        'model_config': model_configs.xgboost_base_train,
        'preproc_config': preproc_configs.preproc_config_nocontext
    },

    'xgboost_config_part_test' : {
        'model_config': model_configs.xgboost_base_part_test,
        'preproc_config': preproc_configs.preproc_config_nocontext
    },

    'xgboost_config_full_test' : {
        'model_config': model_configs.xgboost_full_test,
        'preproc_config': preproc_configs.preproc_config_nocontext
    },

    'roberta_config_train': {
      'model_config': model_configs.roberta_base_train,
      'preproc_config': preproc_configs.preproc_config_context
    },

    'roberta_config_test': {
      'model_config': model_configs.roberta_base_test,
      'preproc_config': preproc_configs.preproc_config_context
    }
}

def get_config(config_name: str):
    """
    returns the configuration with the given name
    :param config_name: name of the configuration
    :return: configuration
    """
    return configurations[config_name]