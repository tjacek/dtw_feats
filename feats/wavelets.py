import numpy as np
import pywt,feat.glob

class SpeedFeature(object):
    def __init__(self,n,wavelet='db1'):
        self.n=n
        self.wavelet=wavelet

    def __call__(self,feature_i):
        aprox=pywt.dwt(feature_i,self.wavelet)[0]
        return feat.glob.simple_smoothnes(feature_i)

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