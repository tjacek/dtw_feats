import numpy as np
import deep,deep.convnet

def get_dataset(in_path,preproc):
    X,y=deep.read_dataset(in_path)
    n_cats=deep.count_cats(y)
    X=np.array([preproc(x_i) for x_i in X])
    return X,y,n_cats

preproc=deep.ImgPreproc(2)
X,y,n_cats=get_dataset('mra/proj',preproc)


print(len(y))
print(n_cats)
print(X.shape)