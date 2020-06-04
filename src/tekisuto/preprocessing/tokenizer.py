"""
Simple preprocessor for tokenization with Gensim
"""
from gensim.utils import simple_preprocess


class Tokenizer:
    def __init__(self, deacc=False):
        self.deacc = deacc

    def sentokenizer(self ,sents):
        """ generator tokenizer for list of strs
        - deacc: default False with punctuation, True without
        """
        assert type(sents) == list, "Input has to be list"

        for sent in sents:
            yield (simple_preprocess(str(sent), deacc=self.deacc))
    
    def doctokenizer(self, docs):
        """ Tokenize list of strings and return list of list of unigrams
        - deacc: default False with punctuation, True without
        """
        assert type(docs) == list, "Input has to be list"

        return list(self.sentokenizer(docs))
