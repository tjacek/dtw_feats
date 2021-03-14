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
				result_i=self.fun(common_i,deep_i,binary=False,clf=clf)[0]
				if(result_i):
					desc=exp_info(common_i,deep_i,result_i)
					lines.append("%s,%s,%s" % desc)
		if(out_path):
			files.save_txt(lines,out_path)
		else:
			print(lines)

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
    dataset="../dtw_paper/MHAD"
    common1=["%s/common/feats/max_z/dtw" % dataset,
            "%s/common/feats/corl/dtw" % dataset,
            "%s/common/feats/skew/dtw" % dataset, 
            "%s/common/feats/std/dtw" % dataset]
    common2="%s/common/MSR_500" % dataset
    binary1="%s/binary/stats/feats" %dataset
    binary2="%s/binary/1D_CNN/feats" %dataset
    common=[None,common1,common2]
    binary=[None,binary1,binary2]
    exp=ExpTemplate()
    exp.standard(common,binary,out_path="MHAD.csv")