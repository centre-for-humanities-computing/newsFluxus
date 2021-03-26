import os
import numpy as np


from change_point import Smoothing


def test_valid_methods():
    """
    check whether all valid methods are indeed valid
    """
    for k in Smoothing().get_valid_methods():
        Smoothing(method=k)


def test_sma():
    """
    test the simple moving average
    """
    ts = np.random.normal(size=10000)
    sma = Smoothing(method="sma", window_size=10)
    smoothed = sma(ts)
    assert np.std(ts) > np.std(smoothed)


def test_gaussian_kernel():
    ts = np.random.normal(size=100)
    gk = Smoothing(method="gaussian_kernel", fwhm=2)
    smoothed = gk(ts)
    assert np.std(ts) > np.std(smoothed)