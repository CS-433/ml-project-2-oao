import string
from nltk.corpus import stopwords
from wordsegment import load, segment
import pickle as pkl
import re
import json


word_dict = pkl.load(open('Data/emnlp_dict.pkl', 'rb'))
load()

# setting up the stopwords list
negation_words = [
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
    "hardly", "scarcely", "barely", "rarely", "seldom", "little", "don't", "can't",
    "won't", "isn't", "aren't", "couldn't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "shouldn't", "wouldn't", "mustn't", "mightn't"
]
stop_words = list((set(stopwords.words('english')) - set(negation_words)).union(set(['<user>', '<url>'])))

# setting up the emoticon dictionary
emoticon_mapping = {
    r':-?\)': ' <happy> ',
    r':-?\(': ' <sad> ',
    r':-?D': ' <big grin> ',
    r':-?P': ' <tongue out> ',
    r';-?\)': ' <wink> ',
    r':\'?\(': ' <crying> ',
    r':-\|': ' <neutral> ',
    r':-\/': ' <skepticism or disapproval> ',
    r':-O': ' <surprise> ',
    r':-\*': ' <kiss> ',
    r'8-\)': ' <cool or sunglasses> ',
    r'<3': ' <love> ',
    r'\^-^': ' <happy or excited> ',
    r'XD': ' <laughing> ',
    r'-_-\'': ' <disapproval or disbelief> ',
    r':v': ' <pac-man> ',
    r'¯\\_(ツ)_/¯': ' <shrug or meh> ',
    
    # Variations with space between face and mouth
    r': -?\)': '<happy>',
    r': -?\(': '<sad>',
    r': -?D': '<big grin>',
    r': -?P': '<tongue out>',
    r'; -?\)': '<wink>',
    r': \'?\(': '<crying>',
    r': -\|': '<neutral>',
    r': -\/': '<skepticism or disapproval>',
    r': -O': '<surprise>',
    r': -\*': '<kiss>',
    r'8 -?\)': '<cool or sunglasses>',
}

# setting up the slang dictionary
with open('ShortendText.json', 'r') as f:
    slang_dict = json.load(f)




def remove_punctuation(text: str) -> str:
    """
    removes all punctuation from a string
    :param text: text to remove punctuation from
    :return: text without punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def identity(text: str) -> str:
    """
    returns the text as it is
    :param text: text to return
    :return: text
    """
    return text

def slang_to_english(text: list) -> list:
    """
    converts slang words to english
    :param text: text to convert
    :return: converted text
    """
    # open the slang json

    # replace slang words with english
    return [slang_dict[word] if word in slang_dict else word for word in text]

def replace_emoticons(text: str) -> str:
    """
    replaces emoticons with their meaning
    :param text: text to replace emoticons in
    :return: text with replaced emoticons
    """
    for emoticon, meaning in emoticon_mapping.items():
        text = re.sub(emoticon, meaning, text)
    return text

def remove_stopwords(text: list) -> list:
    """
    removes all stopwords from a string
    :param text: text to remove stopwords from
    :return: text without stopwords
    """
    return [word for word in text if word not in stop_words]

def to_lowercase(text: str) -> str:
    """
    converts a string to lowercase
    :param text: text to convert to lowercase
    :return: text in lowercase
    """
    return text.lower()

def make_two_consecutive(text : str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def segment_words(text: str) -> list:
    return segment(text)

def correct_word(word: str)->str:
    if word in word_dict:
        return word_dict[word]
    else:
        return word

def replace_hashtag(text: str)-> str:
    return re.sub(r'#', ' talking about ', text)


def remove_numbers(text: str) -> str:
    """
    removes all numbers from a string
    :param text: text to remove numbers from
    :return: text without numbers
    """
    return ''.join([i for i in text if not i.isdigit()])

def remove_whitespace(text: str) -> str:
    """
    removes all whitespaces from a string
    :param text: text to remove whitespaces from
    :return: text without whitespaces
    """
    return ' '.join(text.split())

def remove_single_characters(text: str) -> str:
    """
    removes all single characters from a string
    :param text: text to remove single characters from
    :return: text without single characters
    """
    return ' '.join([word for word in text.split() if len(word) > 1])

def remove_short_words(text: str) -> str:
    """
    removes all words with less than 3 characters from a string
    :param text: text to remove short words from
    :return: text without short words
    """
    return ' '.join([word for word in text.split() if len(word) >= 3])