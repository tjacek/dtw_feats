from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def simple_exp(train,test):
    clf=LogisticRegression()
    clf = clf.fit(train.data, train.target)
    y_true, y_pred = test.target, clf.predict(test.data)
    print(classification_report(y_true, y_pred,digits=4))
