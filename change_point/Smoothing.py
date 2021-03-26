"""
"""
from typing import Optional, Callable, Union
from functools import partial

import numpy as np


class Smoothing:
    """"""

    def __init__(
        self, method: Union[str, None, Callable] = "gaussian_kernel", **kwargs
    ):
        """
        method (Union[str, None, Callable]): Method of smoothing. If None it will simply initalize the class without a method.
        Use Smoothing().get_valid_methods() to see valid methods.
        kwargs: argument to be passed ot the smoothing methods
        """
        self._valid_methods = {"gaussian_kernel": self.gaussian_kernel, "sma": self.sma}

        if method is None:
            self._fun = None
        elif callable(method):
            self._fun = method
            method = method.__name__
        elif method not in self._valid_methods:
            raise ValueError(
                "Method not a valid method, use Smoothing().get_valid_methods() to see valid methods."
            )
        else:
            self._fun = self._valid_methods[method]
        self.kwargs = kwargs
        self.method = method

    def sma(self, ts: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        calculates a simple windowed moving average (SMA)

        ts (list): time series to be smoothed
        window_size (int): the window size to average over
        """
        cumsum, moving_aves = [0], []
        for i, x in enumerate(ts, 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= window_size:
                moving_ave = (cumsum[i] - cumsum[i - window_size]) / window_size
                moving_aves.append(moving_ave)
        return moving_aves

    def gaussian_kernel(self, ts: np.ndarray, sigma: bool = False, fwhm: bool = False):
        """
        gaussian kernel smoother for signal arr
        - sigma: standard deviation of gaussian distribution
        - fwhm: full width at half maximum of gaussian distribution
        """
        y_vals = np.array(ts)
        x_vals = np.arange(ts.shape[0])
        if sigma == fwhm:
            print("[INFO] Define parameters \u03C3 xor FWHM")
        elif fwhm:
            sigma = fwhm / np.sqrt(8 * np.log(2))
        else:
            sigma = sigma
            fwhm = sigma * np.sqrt(8 * np.log(2))

        print(
            "[INFO] Applying Gaussian kernel for \u03C3 = {} and FWHM = {} ".format(
                round(sigma, 2), round(fwhm, 2)
            )
        )

        smoothed_vals = np.zeros(y_vals.shape)
        for x_position in x_vals:
            kernel = np.exp(-((x_vals - x_position) ** 2) / (2 * sigma ** 2))
            kernel = kernel / sum(kernel)
            smoothed_vals[x_position] = sum(y_vals * kernel)

        return smoothed_vals

    def smooth(self, ts, **kwargs):
        if self.kwargs:
            for k in self.kwargs:
                if k not in kwargs:
                    kwargs[k] = self.kwargs[k]
        if self._fun is None:
            raise Exception(
                "Smoothing method is set to None. You should specify a method to call this function"
            )
        return self._fun(ts, **kwargs)

    def __call__(self, ts, **kwargs):
        return self.smooth(ts, **kwargs)

    def get_valid_methods(self):
        return self._valid_methods

    def __str__(self):
        return f"Smoothing(Method={self.method})"
