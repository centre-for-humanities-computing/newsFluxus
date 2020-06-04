"""
Simple dataloader for tabular data 
"""
import os
import numpy as np
import pandas as pd

class DatasetLoaderTable:
    def __init__(self, preprocessors=None):
        self.preprocessors=preprocessors

        if self.preprocessors is None:
            self.preprocessors = list()
    
    def load(self, tablePath, datacol, timecol=None, clscol=None, verbose=-1):
        df = pd.read_csv(tablePath)
        data = df[datacol].values
        if timecol:
            time = df[timecol].values
        else:
            time = []
        if clscol:
            cls = df[clscol].values
        else:
            cls = []
        
        
        for (i, text) in enumerate(data):
            if isinstance(text, str):
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        text = p.preprocess(text)
            else:
                text = text
            
            data[i] = text

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(data)))
            
        
        return (data, time, cls)
