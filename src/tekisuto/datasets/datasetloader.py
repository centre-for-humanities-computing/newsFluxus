"""
"""
import os
import numpy as np

class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors=preprocessors

        if self.preprocessors is None:
            self.preprocessors = list()
    
    def load(self, textPaths, verbose=-1):
        data = list()
        labels = list()
        filenames = list()

        for (i, textPath) in enumerate(textPaths):
            with open(textPath, "r") as f:
                text = f.read()
            label = textPath.split(os.path.sep)[-2]
            filename = textPath.split(os.path.sep)[-1]


            if self.preprocessors is not None:
                for p in self.preprocessors:
                    text = p.preprocess(text)
            
            data.append(text)
            labels.append(label)
            filenames.append(filename)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(textPaths)))
        
        return (data, labels, filenames)
