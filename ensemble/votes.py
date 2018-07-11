import dataset,dataset.instances,utils
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter

class VotingEnsemble(object):
    def __init__(self,norm=True,select=150):
        self.norm=norm
        self.select=select
        self.basic_feats=None

    def __call__(self,basic_paths,deep_path):
        self.basic_feats=get_basic_dataset(basic_paths)
        deep_paths=utils.bottom_files(deep_path)
        all_pred=[]
        y_true=None
        for feat_path_i in deep_paths:
            print(feat_path_i)
            dataset_i=self.get_dataset(feat_path_i)	
            y_i,y_pred=self.get_model(dataset_i)
            if(y_true is None):
                y_true=y_i
            all_pred.append(y_pred)
        y_pred=self.vote(all_pred)
        print(classification_report(y_true, y_pred,digits=4))
       
    def vote(self,all_votes):
        all_votes=np.array(all_votes)
        y_pred=[]
        for vote_i in all_votes.T:
            count =Counter(vote_i)
            cat_i=count.most_common()[0][0]
            y_pred.append(cat_i)
        return y_pred 
    
    def get_dataset(self,feat_path_i):
        adapt_dataset=dataset.read_dataset(feat_path_i)
        full_dataset=dataset.unify_datasets([adapt_dataset,self.basic_feats])
        if(self.norm):
            full_dataset.norm()
        if(not self.select is None ):
            full_dataset.select(self.select)
        return full_dataset

    def get_model(self,dataset_i):
        train,test=dataset_i.split()
        clf=LogisticRegression()
        clf = clf.fit(train.X, train.y)
        y_pred = clf.predict(test.X)
        return test.y,y_pred

def get_basic_dataset(basic_paths):
    datasets=[dataset.read_dataset(basic_i) 
                for basic_i in basic_paths]
    return dataset.unify_datasets(datasets)             