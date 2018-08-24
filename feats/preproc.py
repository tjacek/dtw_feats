import numpy as np

class FeaturePreproc(object):
    def __call__(self,img_seq):
        features=get_features(img_seq)
        new_features=[ self.preproc(feature_i) 
                        for feature_i in features]
        return np.array(new_features).T

    def transform(self,in_path,out_path):
        seqs.io.transform_actions(in_path,out_path,
            transform=self,img_in=False,whole_seq=True)

    def preproc(self):
        raise NotImplementedError()

class ExpSmooth(FeaturePreproc):
    def __init__(self, alpha=0.5):
        self.alpha=alpha

    def preproc(self,feature_i):
        current=feature_i[0]
        smoothed_feature=[current]
        beta=1.0-self.alpha
        for x_i in feature_i[1:]:
            current=self.alpha*x_i + beta*current
            smoothed_feature.append(current)
        return smoothed_feature

class DiffSmooth(FeaturePreproc):

    def preproc(self,feature_i):
        current=feature_i[0]
        diff_feature=[current]
        for x_i in feature_i[1:]:
            diff_feature.append(x_i-current)
            current=x_i
        return diff_feature

def get_features(frames):
    frames=np.array(frames)
    frames=frames.T    
    return frames.tolist()