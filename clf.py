from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def get_cls(clf_type):
    if(clf_type=="SVC"):
        print("SVC")
        return make_SVC()
    else:
        print("LR")
        return LogisticRegression(solver='liblinear')

def make_SVC():
    params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 50,110, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]    
    clf = GridSearchCV(SVC(C=1,probability=True),params, cv=5,scoring='accuracy')
    return clf