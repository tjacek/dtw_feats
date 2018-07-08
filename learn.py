import numpy as np
import deep,deep.convnet,deep.train

def get_dataset(in_path,preproc):
    X,y=deep.read_dataset(in_path)
    y=np.array(y)
    y-= np.min(y)
    n_cats=deep.count_cats(y)
    X=np.array([preproc(x_i) for x_i in X])
    return X,y,n_cats

preproc=deep.ImgPreproc(2)
X,y,n_cats=get_dataset('mra/proj',preproc)
model=deep.convnet.get_model(n_cats,preproc,nn_path='mra/nn')

print(len(y))
print(n_cats)
print(X.shape)
deep.train.train_super_model(X,y,model,num_iter=25)
model.get_model().save('mra/nn')