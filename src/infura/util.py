'''
Infrastructure utility funcitons
'''

import os
import ndjson
import pandas as pd


def resolve_path(path) -> str:
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError('path not found `{}`'.format(path))

# side stuff
def load_ndjson(path):
    with open(path) as fin:
        return ndjson.load(fin)


def validate_preprocessed_data(dobj):
    pass


def validate_lda_input(list_of_docs):
    '''Don't let the user continue if data structure is incompatible.

    Parameters
    ----------
    list_of_docs : list
        Documents / tokens inputed to LDA
    ''' 
    try:
        # is input list?
        isinstance(list_of_docs, list)

        if len(list_of_docs) == 1:
            # if single document is given, first element should be string
            isinstance(list_of_docs[0], str)
        
        else:
            # if multiple documents are given, first element should be a list = document
            isinstance(list_of_docs[0], list)
            # first element of a document should be string = token
            isinstance(list_of_docs[0][0], str)

    except:
        # TODO real error message
        ValueError('Input is not good')