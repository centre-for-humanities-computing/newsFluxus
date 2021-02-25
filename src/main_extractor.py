"""
1. Lemmatize texts
2. Train LDA model
3. Get topic model distribution per document
4. Extract novelty/resonance

python src/main_extractor.py
"""

import os
import pickle

import pandas as pd
import spacy

from tekisuto.preprocessing import CaseFolder
from tekisuto.preprocessing import RegxFilter
from tekisuto.preprocessing import StopWordFilter
from tekisuto.preprocessing import Tokenizer
from tekisuto.models import TopicModel
from tekisuto.models import InfoDynamics
from tekisuto.metrics import jsd

def spacy_lemmatize(texts, nlp, **kwargs):
    docs = nlp.pipe(texts, **kwargs)

    def __lemmatize(doc):
        lemmas = []
        for sent in doc.sents:
            for token in sent:
                lemmas.append(token.lemma_)
        return lemmas

    return [__lemmatize(doc) for doc in docs]

def preprocess_for_topic_models(lemmas: list, lang="da"):
    cf = CaseFolder(lower=True)
    re0 = RegxFilter(pattern=r"\W+")
    re1 = RegxFilter(pattern=r"\d+")
    sw = StopWordFilter(path=os.path.join("res", f"stopwords-{lang}.txt"))
    processors = [cf, re0, re1, sw]
    for processor in processors:
        lemmas = [processor.preprocess(t) for t in lemmas]
    return lemmas


def train_topic_model(tokens, 
                      estimate_topics: bool,
                      tune_topic_range=[10,30,50],
                      plot_topics=False,
                      **kwargs):
    """
    tokens: list of strings (document)
    estimate topics: whether to search a range of topics
    tune_topic_range: number of topics to fit
    plot_topics: quality check, plot coherence by topics
    **kwargs: other arguments to LDAmulticore
    """
    if estimate_topics:
        tm = TopicModel(tokens)
        n, n_cohers = tm.tune_topic_range(
            ntopics=tune_topic_range,
            plot_topics=plot_topics)
        print(f"\n[INFO] Optimal number of topics is {n}")
        tm = TopicModel(tokens)
        tm.fit(n, **kwargs)
    else:
        tm = TopicModel(tokens)
        n = 10
        tm.fit(10, **kwargs)
    return tm, n


def extract_novelty_resonance(df, theta, dates, window):
    idmdl = InfoDynamics(data = theta, time = dates, window = window)
    idmdl.novelty(meas = jsd)
    idmdl.transience(meas = jsd)
    idmdl.resonance(meas = jsd)

    df["novelty"] = idmdl.nsignal
    df["transience"] = idmdl.tsignal
    df["resonance"] = idmdl.rsignal
    df["nsigma"] = idmdl.nsigma
    df["tsigma"] = idmdl.tsigma
    df["rsigma"] = idmdl.rsigma
    return df

if __name__ == '__main__':
    IN_PATH = os.path.join("dat", "speeches_all_metrics.csv")
    OUT_PATH = os.path.join("dat", "speeches_all_metrics_theta.csv")
    ESTIMATE_TOPIPCS = True # whether to tune multiple topic model sizes
    TOPIC_TUNE = [10, 30, 50, 80, 100] # number of topics to tune over in topic model
    PLOT_TOPICS = True # plot a topic of coherence by number of topics
    SAVE_SEMANTIC_TOPICS = True # save the semantic content of the topic model
    WINDOW=3 # window for novelty/resonance
    LANG="da" # language (english = 'en')

    if LANG=="da":
        nlp = spacy.load('da_core_news_lg',  # the language model used
                     disable=["textcat"])
        # you might need to download the model:
        # python -m spacy download da_core_news_lg
    if LANG=="en":
        nlp = spacy.load(nlp = spacy.load("en_core_web_lg"))
        # you might need to download the model:
        # python -m spacy download en_core_web_lg

    # Loading the dataset containing all metrics as non-danish speeches
    # have been removed
    df = pd.read_csv(IN_PATH)
    # sorting date in descending order for correct calculation
    # of novelty and resonance
    df = df.sort_values("date")

    print("\n[INFO] lemmatizing...\n")
    lemmas = spacy_lemmatize(df["text"].tolist(), nlp=nlp)
    lemmas = [' '.join(doc) for doc in lemmas]
    # preprocess
    lemmas = preprocess_for_topic_models(lemmas, lang=LANG)
    # model training
    print("\n[INFO] training model...\n")
    to = Tokenizer()
    tokens = to.doctokenizer(lemmas)
    tm, n = train_topic_model(tokens,
                           ESTIMATE_TOPIPCS,
                           TOPIC_TUNE,
                           PLOT_TOPICS)
    if SAVE_SEMANTIC_TOPICS:
        tm.save_semantic_content(os.path.join("mdl", f"LDA_{n}_topics.txt"))
    # Get topic representation for each document
    print("\n[INFO] Getting topic distribution per document...")
    theta = tm.get_topic_distribution()
    
    out = dict()
    out["model"] = tm.model
    out["id2word"] = tm.dictionary
    out["tokenlists"] = tm.tokenlists
    out["theta"] = theta
    out["dates"] = df['date'].tolist()
    with open(os.path.join("mdl", "speech_topic_dist.pcl"), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    df['theta'] = theta
    ### Extract novelty and resonance
    dates = df["date"].tolist()
    # instantiate and call
    print("[INFO] extracting novelty and resonance...")
    df = extract_novelty_resonance(df, theta, dates, WINDOW)

    df.to_csv(OUT_PATH, index=False)
       
        




    

    

    
    
