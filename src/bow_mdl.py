#!/home/knielbo/virtenvs/teki/bin/python
"""
Driver for preprocessing and training a probabilistic bag-of-words representation representation of newspaper dataset

Parameters:
    - dataset
    - language
    - bytestore
    - estimate: string, "start stop step", range for k number of latent variables grid search
    - sourcename
    - verbose 

EX.
python bow_mdl.py --dataset ../dat/sample.ndjson --language da --bytestore 100 --estimate "20 50 10" --sourcename politiken --verbose 100 

Note:
In case of non-probabilistic representations, the divergence/distance measure has to be change in xyz

"""
import argparse
import os
import json
import pickle
from tekisuto.datasets import DatasetLoaderNdjson
from tekisuto.preprocessing import CaseFolder
from tekisuto.preprocessing import RegxFilter
from tekisuto.preprocessing import StopWordFilter
from tekisuto.preprocessing import Lemmatizer
from tekisuto.preprocessing import Tokenizer
from tekisuto.models import LatentSemantics


def main():    
    # input
    ap = argparse.ArgumentParser(description="[INFO] this is a required preprocessing program for training an newspaper uncertainty model")
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-l", "--language", required=True, help="language of data using ISO 639-1")
    ap.add_argument("-b", "--bytestore", required=False, type=int, default=250, help="frequency for dataloader backup, -1 to deactivate")
    ap.add_argument("-e", "--estimate", required=False, help="estimation mode")
    ap.add_argument("-n", "--sourcename", required=False, default="noname", help="name of the newspaper")
    ap.add_argument("-v", "--verbose", required=False, type=int, default=-1, help="verbose mode (number of object to print), -1 to deactivate")
    args = vars(ap.parse_args())

    # import and preprocess data
    print("\n[INFO] preparing training data ...\n")
    cf = CaseFolder(lower=True)
    re0 = RegxFilter(pattern=r"\W+")
    re1 = RegxFilter(pattern=r"\d+")
    sw = StopWordFilter(path=os.path.join("res", "stopwords-{}.txt".format(args["language"])))
    #sw = StopWordFilter(language="danish")# using NLTK's stopwords
    le = Lemmatizer(lang=args["language"])
    dl = DatasetLoaderNdjson(preprocessors=[cf,re0,re1,sw,le])
    data, _, dates = dl.load(args["dataset"], datesort=True, verbose=args["verbose"], bytestore=args["bytestore"])
    
    # clean up byte stream backup storage from dataloader
    if os.path.isfile("dataloader_bytestorage.pcl"):
        os.remove("dataloader_bytestorage.pcl")

    # model training
    print("\n[INFO] training model...\n")
    to = Tokenizer()
    tokens = to.doctokenizer(data)
    # parameter estimation
    if args["estimate"]:
        print("[INFO] estimating k number of latent variables...")
        print(args["estimate"])
        grid = [int(i) for i in args["estimate"].split()]
        print(grid)
        ls = LatentSemantics(tokens)
        k, _ = ls.coherence_k(krange=list(range(grid[0],grid[1],grid[2])))
        print("[INFO] optimal number of topics: {}".format(k))
        ls = LatentSemantics(tokens, k=k)# TODO: store models 
    # defalut value
    else:
        ls = LatentSemantics(tokens, k=25)# change to your preferred default value
    ls.fit()
    
    # static semantic content for model summary
    print("\n[INFO] writing content to file...\n")
    with open(os.path.join("mdl", "{}_{}_content.txt".format(args["language"], args["sourcename"])), "w") as f:
        for topic in ls.model.show_topics(num_topics=-1, num_words=10):
            f.write("{}\n\n".format(topic))
    
    # theta-based representation
    print("\n[INFO] predicting \u03B8...\n")
    theta = list()
    for i, doc in enumerate(ls.corpus):
        vector = [x[1] for x in ls.model[doc]]
        theta.append(vector)
        if args["verbose"] > 0 and i > 0 and (i + 1) % args["verbose"] == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(data)))

    # serialize model
    print("[INFO] exporting model...")
    out = dict()
    out["model"] = ls.model
    out["id2word"] = ls.id2word
    out["corpus"] = ls.corpus
    out["theta"] = theta
    out["dates"] = dates
    with open(os.path.join("mdl", "{}_{}_model.pcl".format(args["language"], args["sourcename"])), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__=="__main__":
    main()