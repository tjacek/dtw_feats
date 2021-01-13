import numpy as np
from sklearn import neighbors
import ens,learn,exp

def inliner_exp(dtw,deep,binary1,binary2):
	paths=[ (deep,binary1),(deep+dtw,binary1), 
			(deep,[binary1,binary2]),(deep+dtw,[binary1,binary2])]
	results=[]
	for common_i,binary_i in paths:
		basic_i=ens.ensemble(common_i,binary_i,clf="LR",binary=False)
		results.append((False,common_i,binary_i,basic_i))		
		inline_i=inliner_voting(common_i,binary_i,clf="LR")
		results.append((True,common_i,binary_i,inline_i))
	for inline_i,common_i,binary_i,result_i in results:
		desc_common=exp.get_desc(common_i)
		desc_binary=exp.get_desc(binary_i)
		acc_i= result_i.get_acc()
		metric="%.4f,%.4f.%.4f" % result_i.metrics()[:3]
		line_i=(int(inline_i),desc_common,desc_binary,acc_i,metric)
		print("%d,%s,%s,%.4f,%s" % line_i)

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
		print(len(votes))
		votes=np.array(votes)
		pred_i=np.argmax(np.sum(votes,axis=0))
#		print(pred_i)
		y_pred.append(pred_i)
	y_pred=np.array(y_pred)
	return learn.Result(results[0].y_true,y_pred,names)

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
    dataset='3DHOI'
    path='../ICSS_exp/%s' % dataset
    deep=['%s/common/1D_CNN/feats' % path]
    binary1='%s/ens/lstm_gen/feats' % path
    binary2='../ICSS_sim/%s/sim/feats' % dataset
#    binary=['%s/ens/lstm_gen/feats' % path,'../ICSS_sim/%s/sim/feats' % dataset]
    dtw=['%s/dtw/corl/dtw' % path, '%s/dtw/max_z/dtw' % path]
    inliner_exp(dtw,deep,binary1,binary2)
#    result=inliner_voting(deep,binary,clf="LR")
#    result.report()
#    print(result.get_acc())