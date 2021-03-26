"""
This script provide the ChangePoint class

Resources
http://www.claudiobellei.com/2017/01/25/changepoint-bayesian/changepoint_bayesian.py
"""

from typing import Optional, Union

import numpy as np

from .Smoothing import Smoothing

# from bayesian_cp import bayesian_cp

class ChangePoint:
    """
    The ChangePoint detection


    Intended use case:
    pelt_smoothed = ChangePoint(cp="PELT", smoothing="gaussian_kernel")
    outcome = pelt_smoothed.fit(timeseries)

    Results:
    outcome.plot()
    outcome.summary()
    type(outcome)
    PELTOutput

    bayes = pelt_smoothed(cp="bayesian")
    bayes.fit(timeseries, change_points=None)
    """

    def __init__(self, cp: str = "PELT", smoothing: Optional[str] = "gaussian_kernel"):
        """
        cp (str): Methods for change point detection
        smoothing (Optional[str]): Methods for smoothing time series if None it uses no method.
        """
        self.cp_method = cp
        self.smoothing_method = smoothing
        pass

    def bayesian(self, change_points: Optional[int], priors):
        """
        change_points (Optional[int]): Number of change points. If None then the number of change point is estimated
        """
        pass

    def PELT(self):
        pass

    def fit(self, ts: Union[list, np.ndarray]):
        pass

    def __call__(self):
        self.estimate()
        pass

