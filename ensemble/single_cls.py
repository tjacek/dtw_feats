import dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def simple_exp(in_path):
    full_dataset=dataset.read_dataset(in_path)
    full_dataset.norm()
    train,test=full_dataset.split()
    clf=LogisticRegression()
    clf = clf.fit(train.X, train.y)
    y_true, y_pred = test.y, clf.predict(test.X)
    print(classification_report(y_true, y_pred,digits=4))
