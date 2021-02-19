import os.path
import files,ens,learn

class SimpleExp(object):
	def __init__(self,fun=None):
		if(not fun):
			fun=ens.ensemble
		self.fun=fun

	def __call__(self,common_path,binary_path,clf="LR",cf_path=None):
		result=self.fun(common_path,binary_path,clf)
		result.report()
		print(result.get_acc())
		if(cf_path):
			result.get_cf(cf_path)

class ExpTemplate(object):
	def __init__(self,fun=None):
		if(not fun):
			fun=ens.ensemble
		self.fun=fun

	def standard(self,common,binary,out_path=None,clf="LR"):
		paths=files.iter_product([common,binary])   #list(zip(common,binary))
		self(paths,out_path,clf)

	def __call__(self,paths,out_path,clf="LR"):
		lines=[]
		for common_i,deep_i in paths: 
			if(common_i or deep_i):
				result_i=self.fun(common_i,deep_i,clf=clf)[0]
				if(result_i):
					desc=exp_info(common_i,deep_i,result_i)
					lines.append("%s,%s,%s" % desc)
		if(out_path):
			files.save_txt(lines,out_path)
		else:
			print(lines)

#def show_result(in_path,out_path=None,hard=None):
#	lines=[]
#	print("common,ensemble,clf,n_clf,Hard voting,accuracy,precision,recall,fscore")
#	for path_i in files.top_files(in_path):
#		votes_i=learn.read(path_i)
#		if(hard is None):
#			for hard_i in [True,False]:
#				lines.append(show_single(path_i,votes_i,hard_i))
#		else:
#				lines.append(show_single(path_i,votes_i,hard))
#	if(out_path):
#		files.to_csv(lines,out_path)

#def show_single(path_i,votes_i,binary_i):
#	result_i=votes_i.voting(binary_i)
#	metrics_i=result_i.metrics()
#	metrics_i= [result_i.get_acc()] +list(metrics_i[:3])
#	metrics_i="%.4f,%.4f,%.4f,%.4f" % tuple(metrics_i)
#	prefix_i=path_i.split('/')[-1]#.replace("_",",")
#	line_i="%s,%d,%s,%s" % (prefix_i,len(votes_i),str(binary_i),metrics_i)
#	print(line_i)
#	return line_i

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
		desc=[get_name(common_i) for common_i in common_path]
		return "-".join(desc)
	return get_name(common_path)

def get_name(path_i):
	return "_".join(path_i.split("/")[-2:])

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
    common1=["../3DHOI_set2/common/feats/max_z/dtw",
            "../3DHOI_set2/common/feats/corl/dtw",
            "../3DHOI_set2/common/feats/skew/dtw", 
            "../3DHOI_set2/common/feats/std/dtw"]
    common2="../3DHOI_set2/s_dtw"
    binary1="../3DHOI_set2/binary/1D_CNN"
    binary2="../3DHOI_set2/binary/stats"
    common=[None,common1,common2]
    binary=[None,binary1,binary2]
    exp=ExpTemplate()
    exp.standard(common,binary,out_path="test.csv")