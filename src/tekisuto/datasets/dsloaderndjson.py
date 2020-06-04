"""
Dataset loader for ndjson newpaper file 
"""
import os
import json
import pickle

class DatasetLoaderNdjson:
    def __init__(self, preprocessors=None):
        self.preprocessors=preprocessors

        if self.preprocessors is None:
            self.preprocessors = list()
    
    def load(self, filepath, datesort=False, verbose=-1, bytestore=-1):
        """
            - bytestore: integer step for temporary byte stream storage of output
                - option is relevant when using slow prerocessing on larger data sets (e.g., lemmatization)
                - bs backups are not sorted
        """
        data = list()
        titles = list()
        dates = list()

        with open(filepath, "r") as fobj:
            lignes = fobj.readlines()

            for (i, ligne) in enumerate(lignes):
                dobj = json.loads(ligne)
                text = dobj["text"]
                title = dobj["title"]
                date = dobj["date"]

                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        text = p.preprocess(text)
            
                data.append(text)
                titles.append(title)
                dates.append(date)

                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1, len(lignes)))
                
                if bytestore > 0 and i > 0 and (i + 1) % bytestore == 0:
                    print("[INFO] storing intermediary results ...")
                    bsobj = dict()
                    bsobj["data"] = data
                    bsobj["titles"] = titles
                    bsobj["dates"] = dates
                    with open("dataloader_bytestorage.pcl", "wb") as f:
                        pickle.dump(bsobj, f, protocol=pickle.HIGHEST_PROTOCOL)

        if datesort:
            data = [text for _,text in sorted(zip(dates, data))]
            titles = [title for _,title in sorted(zip(dates, titles))]
            dates = sorted(dates)
        
        return (data, titles, dates)
