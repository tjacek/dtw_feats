import learn

def common_errors(paths):
	results=[ learn.read(path_i) for path_i in paths]
	errors=[get_names(result_i) for result_i in results]
	return errors[0].intersection(*errors)

def get_names(result_i):
	return set( [name_i for x,y,name_i in result_i.get_errors()])

paths=["results/max_z","results/skew"]
print(len(common_errors(paths)))