import learn,files,ens

def from_file(paths):
	if(type(paths)==str):
		paths=files.top_files(paths)
	results=[ learn.read(path_i) for path_i in paths]
	return common_errors(results)

def from_ensemble(common_path,binary_path,clf="LR"):
	votes=ens.make_votes(common_path,binary_path,clf,read=None)
#	votes=votes.voting(False)
	return common_errors(votes.results)

def common_errors(results):
	errors=[get_names(result_i) for result_i in results]
	return errors[0].intersection(*errors)

def get_names(result_i):
	return set( [name_i for x,y,name_i in result_i.get_errors()])

#paths=["results/max_z","results/skew"]
binary="../clean3/agum/ens/basic/feats"
print(from_ensemble("s_dtw",binary))