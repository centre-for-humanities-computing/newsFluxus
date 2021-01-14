#!/home/knielbo/virtenvs/teki/bin/python
"""
Driver for extracting the uncertainty model, i.e. time-dependent signal from probabilistic bag-of-words representation representation of newspaper dataset

Parameters:
    - model path to model trained with bow_mdl.py
    - window: int, window to compute novelty and resonance over (for newspapers 7 days)
"""
import argparse
import os
import pickle
import newlinejson
import re
from tekisuto.models import InfoDynamics
from tekisuto.metrics import jsd


def main():
    # input
    ap = argparse.ArgumentParser(description="[INFO] signal extraction for the uncertainty model")
    ap.add_argument("-m", "--model", required=True, help="path to serialized input model")
    ap.add_argument("-w", "--window", required=False, type=int, default=3, help="window to compute novelty and resonance over")
    args = vars(ap.parse_args())

    # import data
    print("[INFO] reading model...")
    with open(args["model"], "rb") as fobj:
        mdl = pickle.load(fobj)
    theta = mdl["theta"]
    time = mdl["dates"]
    
    # instantiate and call
    print("[INFO] extracting signal...")
    idmdl = InfoDynamics(data = theta, time = time, window = args["window"])
    idmdl.novelty(meas = jsd)
    idmdl.transience(meas = jsd)
    idmdl.resonance(meas = jsd)
    
    # export to ndjson
    fname =  os.path.join(
        "mdl", re.sub(
            "model.pcl", "signal.json", format(
                os.path.basename(args["model"])
            )
        )
    )

    if os.path.isfile(fname):
        pass
    else:
        os.mknod(fname)
    print("[INFO] exporting signal to {}".format(fname))
    
    lignes = list()
    for i, date in enumerate(time):
        d = dict()
        d["date"] = date
        d["novelty"] = idmdl.nsignal[i] 
        d["transience"] = idmdl.tsignal[i]
        d["resonance"] = idmdl.rsignal[i]
        d["nsigma"] = idmdl.nsigma[i]
        d["tsigma"] = idmdl.tsigma[i]
        d["rsigma"] = idmdl.rsigma[i]
        lignes.append(d)
    
    with open(fname, "r+") as f:
        newlinejson.dump(lignes, f)
        
if __name__=="__main__":
    main()



