import numpy as np
import pywt

class WaveletPreproc(object):
    def __init__(self,det=0,wavelet_type='db1'):
        self.det=int(det)
        self.wavelet_type = wavelet_type
    
    def __call__(self,feature_i):
        print(feature_i)	
        return pywt.dwt(feature_i,self.wavelet_type)[self.det]

def test_ts(n=25,noise_scale=0.1):
    x=np.arange(n)
    noise=noise_scale*np.random.rand(n)
    return np.sin(x)+noise

preproc=WaveletPreproc()
print(preproc(test_ts()))