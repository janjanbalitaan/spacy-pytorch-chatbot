from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

def snowball_stemmer(word):
    return stemmer.stem(word)