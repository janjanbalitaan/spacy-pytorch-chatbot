from .stemmers import snowball_stemmer

import numpy as np

def bag_of_words(tokenized_sentence, words):
    sentence_words = [snowball_stemmer(word) for word in tokenized_sentence]

    bog = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in sentence_words:
            bog[idx] = 1

    return bog