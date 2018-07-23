import dataset
from ensemble.votes import vote

class MultiAlgEnsemble(object):
    def __init__(self):
        self.algs=[]

    def __call__(self,dataset_path):
        single_dataset=dataset.read_dataset(dataset_path) 
        results=[ alg_i(single_dataset) 
                    for alg_i in self.algs]
        y_true=result_i[0][0]
        all_preds=[result_i[1] for result_i in results]
        y_true=vote(all_preds)