import string
from nltk.corpus import stopwords
from wordsegment import load, segment
import pickle as pkl
import re


word_dict = pkl.load(open('word_dict.pkl', 'rb'))
load()

negation_words = [
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
    "hardly", "scarcely", "barely", "rarely", "seldom", "little", "don't", "can't",
    "won't", "isn't", "aren't", "couldn't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "shouldn't", "wouldn't", "mustn't", "mightn't"
]







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


def remove_stopwords(text: str) -> str:
    """
    removes all stopwords from a string
    :param text: text to remove stopwords from
    :return: text without stopwords
    """
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])


#remove stopwords if they are not negation words and remove <url> and <user>
def remove_stopwords_non_neg(words: list)-> list:
    return [word for word in words if (word not in stopwords.words('english') or word in negation_words) and word not in ['<user>', '<url>']]

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
    return re.sub(r'#', 'talking about ', text)


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