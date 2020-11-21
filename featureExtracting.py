import numpy as np
import pyeeg
import nolds
from pywt import wavedec
import hfda
import eeglib
from entropy import *
import nolds as nolds
from scipy import special
from scipy import stats

class FeatureExtracting:

    """
    This Class has been constructed based on the fact that in DEAP dataset,
    3 seconds baseline has been removed. Hence length of the data recordings is 60 seconds.
    and sampling rate of the data also is 128 Hz.
    Attention: data variable should be fed such as each channel to one function
    """

    def __init__(self, fs=128, ts=60):

        # Removing Baseline and Cutting Data into t seconds segments

        self.fs = fs                    # sampling rate of DEAP dataset [Hz]
        self.ts = ts                    # length of each recording [second]

    def FixedSizedMovingWindow(self, data, segLength=4, segOverlap=2):

        winPoint  = segLength  * self.fs
        overPoint = segOverlap * self.fs
        shape = (data.size - winPoint + 1, winPoint)
        #print("shape",shape)
        stride = data.strides * 2
        #print("strides",stride)
        win = np.lib.stride_tricks.as_strided(data, strides=stride, shape=shape)[0::winPoint - overPoint]
        #print(win.shape) 
        return win

    def StatisticalFeatures(self, data):

        mean  = np.mean(data)               # Mean of data
        std   = np.std(data)                # std of data
        pfd   = pyeeg.pfd(data)             # Petrosian Fractal Dimension
        hurst = pyeeg.hurst(data)           # Hurst Exponent Feature
        dfa   = pyeeg.dfa(data)             # Detrended Fluctuation Analysis
        corr  = nolds.corr_dim(data,1)      # Correlation Dimension Feature
        power = np.sum(np.abs(data)**2)/len(data) # Power feature
        FD    = hfda(data,5)                # fractal dimension
        

        statistics = {"mean":mean, "std":std, "pfd":pfd, "hurst":hurst, "hjorth":hjorth, "dfa":dfa, "corr":corr,
                      "power":power}

        return (statistics)

    def SpectralFeatures(self, data, axis=None):
        ### Hilbert-huang spectrum
        pass


    def DWT(self, data, wavelet="db8", mode="symmetric", level=4,  axis=-1):
        _, coef = wavedec(data, wavelet=wavelet, mode=mode, level=level, axis=axis)
        return (coef)

    def peak2peak(self, data, axis=-1):
        high = np.max(data, axis=axis)
        low = np.min(data, axis=axis)
        return (high - low)

    def meanSquare(self, data, axis=-1):
        temp = data ** 2
        return (np.mean(temp, axis=axis))

    def variance(self, data, axis=-1):
        var = (np.std(data, axis=axis)) ** 2
        return var

    def hjorth(self, data, axis=-1):
        activity = eeglib.features.hjorthActivity(data)
        mobility = eeglib.features.hjorthMobility(data)
        complexity = eeglib.features.hjorthComplexity(data)

        return activity, mobility, complexity

    def approxEntropy(self, data):
        temp = app_entropy(data, order=2, metric='chebyshev')
        return temp

    def correlationDimension(self, data):
        temp = nolds.corr_dim(data, emb_dim=3)
        return temp

    def Kolmogolov(self,data):
        temp = special.kolmogorov(data)
        return temp

    def Lyapunov(self, data):
        temp = nolds.lyap_e(data, emb_dim=-1)
        return temp

    def permutationEntropy(self, data):
        temp = perm_entropy(data, order=3, normalize=True)
        return temp

    def singularSpectropyEntropy(self, data):
        temp = svd_entropy(data, order=3, normalize=True)
        return temp

    def spectralEntropy(self, data):
        temp = spectral_entropy(data, 128, normalize=True)
        return temp

    def shannonEntropy(self, data):
        temp = stats.entropy(data)
        return temp
    










