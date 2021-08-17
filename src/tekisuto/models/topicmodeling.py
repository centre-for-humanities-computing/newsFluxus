"""
Topic model class
"""
import os
import numpy as np

import matplotlib.pyplot as plt

from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel

class TopicModel():
    """
    Shamelessly stolen/slightly modified from TextToX
    """
    def __init__(self, tokenlists):
        self.tokenlists = tokenlists
    
    def fit(
        self,
        num_topics,
        no_below=1,
        no_above=0.9,
        keep_n=None,
        keep_tokens=None,
        remove_most_freq_n=None,
        bad_tokens=None,
        model="ldamulticore",
        bigrams=True,
        **kwargs,
        ):
        """
        no_below (int|None) – Keep tokens which are contained in at least
        no_below documents.
        no_above (float|None): Keep tokens which are contained in no
        more than no_above documents (fraction of total corpus size,
        not an absolute number).
        keep_n (int|None) – Keep only the first keep_n most frequent
        tokens.
        keep_tokens (iterable of str) – Iterable of tokens that must stay in
        dictionary after filtering.
        remove_most_freq_n (int|None): Remove n most frequent tokens
        model ('ldamulticore'|'lda'|'ldamallet')
        """
        if bigrams:
            phrases = models.Phrases(self.tokenlists, delimiter=b" ")
            phraser = models.phrases.Phraser(phrases)
            self.tokenlists = [phraser[tl] for tl in self.tokenlists]

        dictionary = corpora.Dictionary(self.tokenlists)
        self.dictionary = dictionary

        if remove_most_freq_n:
            dictionary.filter_n_most_frequent(remove_most_freq_n)
        dictionary.filter_extremes(
            no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens
        )

        bows = [dictionary.doc2bow(tl) for tl in self.tokenlists]
        self.bows = bows

        if bad_tokens:
            dictionary.filter_tokens(
                bad_ids=[dictionary.id2token[tok] for tok in bad_tokens]
            )
        self.bows = bows
        self.dictionary = dictionary
        if model == "ldamulticore":       
            self.model = models.LdaMulticore(
                bows, num_topics=num_topics, id2word=dictionary, **kwargs
            )
        if model == "lda":
            self.model = models.LdaModel(
                bows, num_topics=num_topics, id2word=dictionary, **kwargs
            )

    def tune_topic_range(self, ntopics=[10,20,30,40,50], plot_topics=False, **kwargs):
        n_cohers = list()
        for i, n in enumerate(ntopics):
            model_path = os.path.join("mdl", f"LDA_{n}_topics")
            print("[INFO] Estimating coherence model for {} topics, iteration {}".format(n, i))
            tm = TopicModel(self.tokenlists)
            tm.fit(n, **kwargs)
            n_cohers.append(tm.get_coherence())
            # figure out a clever way to save/load with bow and dictionary
            #tm.save_model(model_path)
        
        n_cohers = np.array(n_cohers, dtype=np.float)
        idx =  n_cohers.argsort()[-len(ntopics):][::-1]
        n = ntopics[idx[np.argmax(n_cohers[idx]) & (np.gradient(n_cohers)[idx] >= 0)][0]]
        if plot_topics:
            plt.plot(ntopics, n_cohers)
            plt.xlabel("Number of topics")
            plt.ylabel("Coherence")
            plt.savefig("topics.png")
            plt.close()
        return n, n_cohers

    def get_coherence(self, **kwargs):
            coherence_model_lda = CoherenceModel(
                model=self.model, texts=self.tokenlists, corpus=self.bows, **kwargs
            )
            return coherence_model_lda.get_coherence()

    def get_log_complexity(self):
        return self.model.log_perplexity(self.bows)

    def get_lda(self):
        return self.lda

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = models.LdaModel.load(path)

    def save_semantic_content(self, path):
        with open(path, "w") as f:
            for topic in self.model.show_topics(num_topics=-1, num_words=10):
                f.write("{}\n\n".format(topic))

    def get_topic_distribution(self):
        theta = list()
        for bow in self.bows:
            thetas = [x[1] for x in 
                      self.model.get_document_topics(bow, 
                        minimum_probability=0.0)]
            theta.append(thetas)
        return theta
