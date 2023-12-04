from TextProc.helper import *


preproc_config_1 = [
    identity
]

preproc_config_2 = [
    remove_punctuation,
    remove_stopwords,
    to_lowercase,
    remove_numbers,
    remove_whitespace
]

preproc_config_nocontext = [
    replace_emoticons,
    make_two_consecutive,
    replace_hashtag,
    string_to_list,
    slang_to_english,
    correct_word,
    remove_stopwords,
    lemmatize_word,
    list_to_string
]

preproc_config_context = [
    replace_emoticons, 
    make_two_consecutive,
    replace_hashtag,
    string_to_list,
    slang_to_english,
    correct_word,
    list_to_string
]