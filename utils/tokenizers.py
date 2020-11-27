import spacy

nlp = spacy.load("zh_core_web_sm")

def spacy_tokenizer(text):
    doc = nlp(text)
    return [str(d).lower() for d in doc]