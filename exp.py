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
		result=ens.ensemble(common_path,binary_path,binary=False,clf=clf_i)
		result.save(out_i)

def show_result(in_path):
	print("common,binary,Hard,clf,Hard voting,accuracy,precision,recall,fscore")
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
	prefix_i=path_i.split('/')[-1]#.replace("_",",")
	line_i="%s,%s,%s" % (prefix_i,str(binary_i),metrics_i)
	print(line_i)

def get_desc(common_path):
	if(type(common_path)==list):
		desc=[common_i.split("/")[-2] for common_i in common_path]
		return "-".join(desc)
	return common_path.split("/")[-2]

def find_path(in_path):
	paths=files.bottom_files(in_path,full_paths=True)
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

#common_path="good/ae_basic"
#binary_path="good/ens"

dtw=["../ICSS_exp/MSR/dtw/max_z/person","../ICSS_exp/MSR/dtw/corl/person"]
common_path="../ICSS_exp/MSR/common/stats/feats"
binary_path="../ICSS_exp/MSR/ens/lstm_gen/feats"
#dtw.append(common_path)
#ens_exp(dtw,binary_path,"MSR5")
#
dtw=['../ICSS_exp/3D_HOI/dtw/corl/dtw','../ICSS_exp/3D_HOI/dtw/max_z/dtw']
dtw1=['../ICSS_exp/3D_HOI/dtw/corl/dtw','../ICSS_exp/3D_HOI/dtw/max_z/dtw','../ICSS_exp/3D_HOI/common/1D_CNN_AE/feats']
common=['../ICSS_exp/3D_HOI/common/1D_CNN_AE/feats',dtw,dtw1]

#multi_exp(common,"../ICSS_exp/3D_HOI/ens/lstm/feats","test")
#find_path("../ICSS_exp/3D_HOI")
#ens_exp(common_path,binary_path,"test")
show_result("test")