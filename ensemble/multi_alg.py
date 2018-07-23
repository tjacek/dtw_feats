import sklearn,sklearn.ensemble
import dataset
from ensemble.votes import vote
import ensemble.single_cls

class MultiAlgEnsemble(object):
    def __init__(self):
        self.algs=[svm_alg,random_forest_alg]

    def __call__(self,dataset_path):
        single_dataset=dataset.read_dataset(dataset_path) 
        results=[ alg_i(single_dataset) 
                    for alg_i in self.algs]
        y_true=results[0][0]
        all_preds=[result_i[1] for result_i in results]
        y_pred=vote(all_preds)
        ensemble.single_cls.show_result(y_true,y_pred)

def svm_alg(dataset_i):
    train,test=dataset_i.split()
    clf = sklearn.svm.SVC()
    clf.fit(train.X, train.y)
    y_pred = clf.predict(test.X)
    return test.y,y_pred

def random_forest_alg(dataset_i):
    train,test=dataset_i.split()
    clf = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(train.X, train.y)
    y_pred = clf.predict(test.X)
    return test.y,y_pred