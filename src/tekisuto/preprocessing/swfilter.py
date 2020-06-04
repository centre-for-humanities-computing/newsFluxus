"""
Simple preprocesser for stopword filtering

path: path to custom list
token: if text is tokenized
language: filter lanugage is path is False
"""
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktLanguageVars

class StopWordFilter:
    def __init__(self, path=False, token=False, language="english"):
        self.path = path
        self.token = token
        self.lang = language
    
    def preprocess(self, text):
        if self.path:
            with open(self.path, "r") as f:
                sw = set(f.read().split("\n")[:-1])
        else:
            sw = set(stopwords.words(self.lang))
        
        if self.token:
            unigrams = text
        else:
            plv = PunktLanguageVars()
            unigrams = plv.word_tokenize(text.lower())
        
        return " ".join([unigram for unigram in unigrams if not unigram in sw])

