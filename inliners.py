import numpy as np
from sklearn import neighbors
import ens,learn

def inliner_voting(common,deep,clf="LR"):
	datasets=ens.read_dataset(None,deep)
	inliners=[ get_inliner_dict(date_i) for date_i in datasets]
	full_datasets=ens.read_dataset(common,deep)
	results=learn.train_ens(full_datasets,clf=clf)
	weights=[inliner_weights(inliners_i,result_i)
				for inliners_i,result_i in zip(inliners,results)]
	names=results[0].names
	n_clf=len(results)
	y_pred=[]
	for j,name_j in enumerate(names):
		print(name_j)
		weights_j=[weights[i][j] for i in range(n_clf)]
		if(sum(weights_j)>2):
			votes=[results[i].y_pred[j] for i in range(n_clf)
						if(weights_j[i]==1)]
		else:
			votes=[results[i].y_pred[j] for i in range(n_clf)]
#		print(weights_j)
#		print(len(votes))
		votes=np.array(votes)
		pred_i=np.argmax(np.sum(votes,axis=0))
#		print(pred_i)
		y_pred.append(pred_i)
	y_pred=np.array(y_pred)
	return learn.Result(results[0].y_true,y_pred,names)
#	print(y_pred)

def get_inliner_dict(date_i,k=3):
	train,test=date_i.split()
	X_train,y_train,train_names=train.as_dataset()
	clf_i= neighbors.KNeighborsClassifier(k)
	clf_i.fit(X_train,y_train)
	X_test,y_test,test_names=test.as_dataset()
	result=clf_i.predict(X_test)
	return {name_j:result[j] 
    			for j,name_j in enumerate(test_names)}

def inliner_weights(inliner_i,result_i):
	y_true,y_pred=result_i.as_labels()
	weights=[ int(y_i==inliner_i[name_j]) 
	  			for y_i,name_j in zip(y_pred,result_i.names)]
	return weights

if __name__ == "__main__":
    dataset='../ICSS_exp/3DHOI/'
    deep=['%s/common/1D_CNN/feats' % dataset]
    binary='%s/ens/lstm/feats' % dataset
    dtw=['%s/dtw/corl/dtw' % dataset, '%s/dtw/max_z/dtw' % dataset]
    result=inliner_voting(dtw+deep,binary,clf="LR")
    print(result.get_acc())