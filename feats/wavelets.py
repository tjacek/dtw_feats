import numpy as np
import pywt

def test_ts(n=25,noise_scale=0.1):
    x=np.arange(n)
    noise=noise_scale*np.random.rand(n)
    return np.sin(x)+noise

result=pywt.dwt(test_ts(), 'db1')
print(test_ts())
print(result[0])
print(result[1])
	