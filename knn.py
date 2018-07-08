from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix
import graph

def knn(nn_graph,k):
    names=nn_graph.names()
    true_y=[nn_graph.get_cat(name_i) 
                for name_i in names]
    pred_y=[ pred_cat(name_i,nn_graph,k) 
                for name_i in names]
    return pred_y,true_y

def pred_cat(name_i,nn_graph,k):
    neighbors=nn_graph.get_neighbors(name_i,n=k)
    if(k==0):
        return neighbors[0]
    else:
        count =Counter(neighbors)
        return count.most_common()[0][0]

def check_prediction(pred_y,true_y):
    print(classification_report(pred_y,true_y,digits=4))
    print(confusion_matrix(pred_y,true_y))

def get_accuracy(matrix):
    return np.trace(matrix)/np.sum(matrix)

if __name__ == "__main__":
    nn_graph=graph.make_nn_graph("mhad/skew_pairs","mhad/skew")
    pred_y,true_y=knn(nn_graph,k=10)
    check_prediction(pred_y,true_y)