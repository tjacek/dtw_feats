import os.path
import files,ens,learn

class ExpTemplate(object):
	def __init__(self,fun=None):
		if(not fun):
			fun=ens.make_votes
		self.fun=fun

	def __call__(self,dtw,deep,binary,out_path,clf="LR"):
		paths=files.iter_product([[dtw,deep], [binary,None]])
		paths.append((dtw+deep,binary))
		paths.append((None,binary))
		lines=[]
		for common_i,deep_i in paths: 
			votes=self.fun(common_i,deep_i,clf=clf)
			result_i=votes.voting(False)
			desc=exp_info(common_i,deep_i,result_i)
			lines.append("%s,%s,%s" % desc)
		files.to_csv(lines,out_path)

def single_exp(common_path,binary_path,out_path,fun=None,clf="LR"):
	if(not fun):
		fun=ens.make_votes
	if(type(clf)==str):
		clf=[clf]
	if(out_path):
		files.make_dir(out_path)
	lines=[]
	for clf_i in clf:
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

def show_result(in_path,out_path=None,hard=None):
	lines=[]
	print("common,ensemble,clf,n_clf,Hard voting,accuracy,precision,recall,fscore")
	for path_i in files.top_files(in_path):
		votes_i=learn.read(path_i)
		if(hard is None):
			for hard_i in [True,False]:
				lines.append(show_single(path_i,votes_i,hard_i))
		else:
				lines.append(show_single(path_i,votes_i,hard))
	if(out_path):
		files.to_csv(lines,out_path)

def show_single(path_i,votes_i,binary_i):
	result_i=votes_i.voting(binary_i)
	metrics_i=result_i.metrics()
	metrics_i= [result_i.get_acc()] +list(metrics_i[:3])
	metrics_i="%.4f,%.4f,%.4f,%.4f" % tuple(metrics_i)
	prefix_i=path_i.split('/')[-1]#.replace("_",",")
	line_i="%s,%d,%s,%s" % (prefix_i,len(votes_i),str(binary_i),metrics_i)
	print(line_i)
	return line_i

def exp_info(common_i,binary_i,result_i):
	desc_common=get_desc(common_i)
	desc_binary=get_desc(binary_i)
	acc_i= result_i.get_acc()
	metrics="%.4f,%.4f,%.4f" % result_i.metrics()[:3]
	metrics="%.4f,%s" % (acc_i,metrics)
	return desc_common,desc_binary,metrics

def get_desc(common_path):
	if(common_path is None):
		return "-"
	if(type(common_path)==list):
		desc=[common_i.split("/")[-2] for common_i in common_path]
		return "-".join(desc)
#	raise Exception(common_path)
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
	return common_paths,binary_paths

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

if __name__ == "__main__":
	deep=['../ICSS_exp/3DHOI/common/1D_CNN/feats']#,'../ICSS_exp/3DHOI/common/stats/feats']
	binary='../ICSS_exp/3DHOI/ens/lstm/feats'
	dtw=['../ICSS_exp/3DHOI/dtw/corl/person', '../ICSS_exp/3DHOI/dtw/max_z/person']
	dtw_exp=ExpTemplate()
	dtw_exp(dtw,deep,binary,"3DHOI.csv")
#	show_result("3DHOI",hard=False)#,"reduction/SVC.csv")