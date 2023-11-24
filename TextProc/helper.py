import string
from nltk.corpus import stopwords

def remove_punctuation(text: str) -> str:
    """
    removes all punctuation from a string
    :param text: text to remove punctuation from
    :return: text without punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text: str) -> str:
    """
    removes all stopwords from a string
    :param text: text to remove stopwords from
    :return: text without stopwords
    """
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])

def to_lowercase(text: str) -> str:
    """
    converts a string to lowercase
    :param text: text to convert to lowercase
    :return: text in lowercase
    """
    return text.lower()

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