from .constants import get_ignore_words
from .stemmers import snowball_stemmer

def remove_ignore_words(word_list):
    return [snowball_stemmer(word) for word in word_list if word not in get_ignore_words()]

def sort_word_list(word_list):
    return sorted(set(word_list))

def sort_tags(tags):
    return sorted(set(tags))

