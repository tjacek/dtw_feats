import files,ens,learn
#import pickle

def ens_exp(common_path,binary_path,out_path):
	files.make_dir(out_path)
	for clf_i in ["LR","SVC"]:
#		for binary_i in [True,False]:
		result=ens.ensemble(common_path,binary_path,binary=False,clf=clf_i)
		print(result.y_pred[0])
		out_i="%s/%s" % (out_path,clf_i)#,str(binary_i))
		result.save(out_i)

def show_result(in_path):
	print("Hard,clf,Hard voting,accuracy,precision,recall,fscore")
	for path_i in files.top_files(in_path):
		result_i=learn.read(path_i)
		for binary_i in [True,False]:
			if(binary_i):
				show_single(path_i,result_i.as_hard_votes(),binary_i)
			else:
				show_single(path_i,result_i,binary_i)

def show_single(path_i,result_i,binary_i):
	metrics_i=result_i.metrics()
	metrics_i= [result_i.get_acc()] +list(metrics_i[:3])
	metrics_i="%s,%s,%s,%s" % tuple(metrics_i)
	prefix_i=path_i.split('/')[-1].replace("_",",")
	line_i="%s,%s,%s" % (prefix_i,str(binary_i),metrics_i)
	print(line_i)

#common_path="good/ae_basic"
#binary_path="good/ens"

dtw=["../ICSS_exp/MSR/dtw/max_z/person","../ICSS_exp/MSR/dtw/corl/person"]
common_path="../ICSS_exp/MSR/common/stats/feats"
binary_path="../ICSS_exp/MSR/ens/lstm_gen/feats"
dtw.append(common_path)
ens_exp(dtw,binary_path,"MSR5")
show_result("MSR5")