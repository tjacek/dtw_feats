import numpy as np
import seqs.io 
import feats

class Preproc(object):
    def transform(self,in_path,out_path):
        seqs.io.transform_actions(in_path,out_path,
            transform=self,img_in=False,whole_seq=True)

class FeaturePreproc(Preproc):
    def __init__(self,preproc):
        self.preproc=preproc

    def __call__(self,img_seq):
        features=feats.get_features(img_seq)
        new_features=[ self.preproc(feature_i) 
                        for feature_i in features]
        return np.array(new_features).T

class DistancePreproc(Preproc):
    def __call__(self,img_seq):
        mean_i=np.mean(img_seq)
        dist_i=np.array([np.linalg.norm(point_i-mean_i) 
                            for point_i in img_seq])
        dist_i=np.expand_dims(dist_i,1)
        print(dist_i.shape)
        return np.array(dist_i)

class ExpSmooth(object):
    def __init__(self, alpha=0.5):
        self.alpha=alpha

    def __call__(self,feature_i):
        beta=1.0-self.alpha
        return self.alpha*feature_i[1:]+beta*feature_i[:-1]

class DiffPreproc(object):
    def __init__(self,sign_only=False):
        self.sign_only=sign_only

    def __call__(self,feature_i):
        if(self.sign_only):
            return np.sign(np.diff(feature_i))
        return np.diff(feature_i)

class FourierSmooth(object):
    def __init__(self, n=5):
        self.n = n

    def __call__(self,feature_i):
        rft = np.fft.rfft(feature_i)
        rft[self.n:] = 0
        return np.fft.irfft(rft)

def fourier_magnitude(feature_i):
    rft = np.fft.rfft(feature_i)
    return np.sqrt(rft*rft.conjugate()).real

def norm_scale(feature_i):
    return (feature_i/np.amax(feature_i))