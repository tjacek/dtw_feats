import os.path
import files,ens,learn
#import pickle

def multi_exp(common_path,binary_path,out_path):
	for common_i in common_path:
		single_exp(common_i,binary_path,out_path)

def single_exp(common_path,binary_path,out_path):
	files.make_dir(out_path)
	for clf_i in ["LR"]:#,"SVC"]:
		desc_i=get_desc(common_path)
		ens_i=binary_path.split("/")[-2]
		out_i="%s/%s,%s,%s" % (out_path,desc_i,ens_i,clf_i)
		print(out_i)
		votes=ens.make_votes(common_path,binary_path,clf=clf_i)#ens.ensemble(common_path,binary_path,binary=False,clf=clf_i)
		votes.save(out_i)
		print(votes.voting().get_acc())

def show_result(in_path):
	print("common,binary,Hard,clf,Hard voting,accuracy,precision,recall,fscore")
	for path_i in files.top_files(in_path):
		result_i=learn.read(path_i)
		for binary_i in [True,False]:
			show_single(path_i,result_i.voting(binary_i),binary_i)

def show_single(path_i,result_i,binary_i):
	metrics_i=result_i.metrics()
	metrics_i= [result_i.get_acc()] +list(metrics_i[:3])
	metrics_i="%s,%s,%s,%s" % tuple(metrics_i)
	prefix_i=path_i.split('/')[-1]#.replace("_",",")
	line_i="%s,%s,%s" % (prefix_i,str(binary_i),metrics_i)
	print(line_i)

def get_desc(common_path):
	if(type(common_path)==list):
		desc=[common_i.split("/")[-2] for common_i in common_path]
		return "-".join(desc)
	return common_path.split("/")[-2]

def find_path(in_path):
	paths=files.all_files(in_path)
	common_paths=[]
	binary_paths=[]
	for path_i in paths:
		if(path_i.split("/")[-1]=="feats"):
			if(os.path.isdir(path_i)):
				binary_paths.append(path_i)
			else:
				common_paths.append(path_i)
	print(common_paths)
	print(binary_paths)

def find_dtw(in_path,dtw_type="dtw"):
	paths=[path_i for path_i in files.all_dicts(in_path)
				if(path_i.split("/")[-1]=="dtw")]
	dtw_feats=[]
	for path_i in paths:
		dtw_i=["%s/%s" % (feat_j,dtw_type) 
				for feat_j in files.top_files(path_i)]
		dtw_feats.append(dtw_i)
	print(dtw_feats)
	return dtw_feats

def dtw_exp(dtw,deep,binary,out_path):
	single_exp(dtw,binary,out_path)
	for deep_i in deep:
		single_exp(deep_i,binary,out_path)
		single_exp(dtw+[deep_i],binary,out_path)

#find_path('../ICSS_exp/MSR')
#find_dtw('../ICSS_exp/MSR')

deep=['../ICSS_exp/MSR/common/1D_CNN/feats', '../ICSS_exp/MSR/common/stats/feats']
binary='../ICSS_exp/MSR/ens/lstm_gen/feats'
dtw=['../ICSS_exp/MSR/dtw/corl/person', '../ICSS_exp/MSR/dtw/max_z/person']
dtw_exp(dtw,deep,binary,"MSR2")
show_result("MSR2")