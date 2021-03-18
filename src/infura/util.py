'''
Infrastructure utility funcitons
'''

import os

import ndjson
import pandas as pd
from rapidfuzz import process, utils

# duplicates
def flag_fuzzy_duplicates(input_list, cutoff=93, validate=False):
    '''Flag indices in input_list, where fuzzy duplicates were found

    Parameters
    ----------
    input_list : list
        List of texts (at least two elements)
    cutoff : int, optional
        Similarity between texts has to be at least `cutoff`,
        to be considered duplicate. By default 93 (out of 100)
    validate : bool, optional
        Also return a list of tuples with matching details?, by default False

    Returns
    -------
    Indices : list of bool
        Fuzzy-duplicate elements are True
    
    Validation: list of tuples, optional
        where tuple[0]: query, [1]: score, [2]: index of duplicate
    '''
    if len(input_list) >= 2:
        # only attempt to flag anything if input is sufficient length

        # list of falses same lengths as input
        indices = [False] * len(input_list)
        
        if validate:
            # track matches
            validation = []

        for (i, processed_query) in enumerate(input_list):
            # list to compare current query to
            choices = input_list.copy()
            # remove current text from choices
            del choices[i]
            # find fuzzy matches
            match = process.extractOne(processed_query, choices, processor=None, score_cutoff=cutoff)
            if match:
                # validation
                if validate:
                    validation.append(match)
                # flag duplicate as True
                # match[0]: query, match[1]: score, match[2]: index of duplicate
                indices[match[2]] = True
            else:
                pass

        if validate:
            return indices, validation
        else:
            return indices


# paths
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