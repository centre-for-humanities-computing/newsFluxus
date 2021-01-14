"""
Simple preprocesser for lemmatization with Stanza
"""
import stanza

class Lemmatizer:
    def __init__(self, lang="en"):
        self.nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma')
    
    def preprocess(self, text):
        try:
            doc = self.nlp(text)
            lemma = [word.lemma for sent in doc.sentences for word in sent.words if word.lemma is not None]
        except:
            lemma = "nan"

        return " ".join(lemma)