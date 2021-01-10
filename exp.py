import os.path
import files,ens,learn
#import pickle

def multi_exp(common_path,binary_path,out_path):
	for common_i in common_path:
		single_exp(common_i,binary_path,out_path)

def single_exp(common_path,binary_path,out_path,fun=None):
	if(not fun):
		fun=ens.make_votes
	files.make_dir(out_path)
	lines=[]
	for clf_i in ["LR"]:#,"SVC"]:
		desc_i=get_desc(common_path)
		ens_i=get_desc(binary_path) 
		out_i="%s/%s,%s,%s" % (out_path,desc_i,ens_i,clf_i)
		print(out_i)
		lines.append(out_i)
		if(not os.path.exists(out_i)):
			votes=fun(common_path,binary_path,clf=clf_i)
			votes.save(out_i)
		else:
			votes=learn.read(out_i)
		print(votes.voting().get_acc())
	return lines

def show_result(in_path,out_path=None):
	lines=[]
	print("common,ensemble,clf,n_clf,Hard voting,accuracy,precision,recall,fscore")
	for path_i in files.top_files(in_path):
		votes_i=learn.read(path_i)
		for binary_i in [True,False]:
			lines.append(show_single(path_i,votes_i,binary_i))
	if(out_path):
		files.to_csv(lines,out_path)

def show_single(path_i,votes_i,binary_i):
	result_i=votes_i.voting(binary_i)
	metrics_i=result_i.metrics()
	metrics_i= [result_i.get_acc()] +list(metrics_i[:3])
	metrics_i="%4f,%4f,%4f,%4f" % tuple(metrics_i)
	prefix_i=path_i.split('/')[-1]#.replace("_",",")
	line_i="%s,%d,%s,%s" % (prefix_i,len(votes_i),str(binary_i),metrics_i)
	print(line_i)
	return line_i

def get_desc(common_path):
	if(common_path is None):
		return "-"
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
	single_exp(None,binary,out_path)
	single_exp(dtw,binary,out_path)
	single_exp(dtw,None,out_path)
	for deep_i in deep:
		single_exp(deep_i,binary,out_path)
		single_exp(dtw+[deep_i],binary,out_path)
		single_exp(deep_i,None,out_path)

if __name__ == "__main__":
	deep=['../ICSS_exp/3DHOI/common/1D_CNN/feats','../ICSS_exp/3DHOI/common/stats/feats']
	binary='../ICSS_exp/3DHOI/ens/lstm/feats'
	dtw=['../ICSS_exp/3DHOI/dtw/corl/person', '../ICSS_exp/3DHOI/dtw/max_z/person']
#	dtw_exp(dtw,deep,binary,"3DHOI")
	show_result("selection/MHAD","selection/MHAD.csv")