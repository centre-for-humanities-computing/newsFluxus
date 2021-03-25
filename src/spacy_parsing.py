"""
"""


def spacy_preprocess(texts: list,
                     nlp, **kwargs
                     ):
    """
    >>> nlp = spacy.load('da_core_news_lg', disable=["textcat"])
    >>> res = spacy_preprocess(
            texts = ["dette er en test text med et navn, nemlig Lasse Hansen"],
            nlp=nlp)
    """
    docs = nlp.pipe(texts, **kwargs)

    def __extract_spacy(doc):
        doc_features = {
            "sent_id": [],
            "token": [],
            "token_character_span": [],
            "lemma": [],
            "pos": [],
            "dep": [],
            "ner": [],
            "noun_chunk": [],
            "noun_chunk_token_span": []}
        for sent_i, sent in enumerate(doc.sents):
            for token in sent:
                doc_features["sent_id"].append(sent_i)
                doc_features["token"].append(token.text)
                doc_features["token_character_span"].append(
                    (token.idx, token.idx+len(token)))
                doc_features["lemma"].append(token.lemma_)
                doc_features["pos"].append(token.pos_)
                doc_features["dep"].append(token.dep_)
                doc_features["ner"].append(token.ent_type_)
            for nc in sent.noun_chunks:
                doc_features["noun_chunk"].append(nc.text)
                doc_features["noun_chunk_token_span"].append(
                    (nc.start, nc.end))
        return doc_features
    return [__extract_spacy(doc) for doc in docs]


if __name__ == "__main__":
    import spacy
    spacy.prefer_gpu()  # enable GPU
    nlp = spacy.load('da_core_news_lg',  # the language model used
                     disable=["textcat"])
    preprocesed = spacy_preprocess(
        ["a list of test texts"], nlp=nlp, n_process=16)  # number of processes
