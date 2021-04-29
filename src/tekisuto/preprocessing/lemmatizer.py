"""
Simple preprocesser for lemmatization with Stanza
"""
import stanza

lang_dict={"da": "da_core_news_lg", "en": "en_core_web_lg"}}

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

class LemmatizerSpacy:
    def __init__(self, lang="en", nlp=None):
        if nlp is None:
            model = lang_dict[lang]
            self.nlp = spacy.load(model)
        else:
            self.nlp = nlp 
    
    def preprocess(self, text):
        doc = self.nlp(text)
        lemma = [t.lemma_ for t in doc]
        return " ".join(lemma)
