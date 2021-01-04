import numpy as np
import feats
from sklearn.metrics import accuracy_score
import clf,files,feats

def train_model(data,binary=False,clf_type="LR",acc_only=False):
    if(type(data)==str):	
        data=feats.read(data)[0]
#    data.norm()
    print(data.dim())
    print(len(data))
    train,test=data.split()
    model= clf.get_cls(clf_type)
    X_train,y_train= train.get_X(),train.get_labels()
    model.fit(X_train,y_train)
    X_test,y_true=test.get_X(),test.get_labels()
    if(binary):
        y_pred=model.predict(X_test)
    else:
        y_pred=model.predict_proba(X_test)
    if(acc_only):
        return accuracy_score(y_true,y_pred)
    else:
        return y_true,y_pred,test.info

def voting(results,binary):
    votes=get_prob(results)
    votes=np.sum(votes,axis=0)
    y_pred=[np.argmax(vote_i) for vote_i in votes]
    return y_pred

def get_prob(results):
    return np.array([result_i[1] for result_i in results])

def get_acc(paths,clf="LR"):
    datasets=tools.read_datasets(paths)
    return [train_model(data_i,True,clf,True) 
                for data_i in datasets]