import numpy as np 
from sklearn.linear_model import LinearRegression

def count_mins(feature_i):
    extr=local_extr(feature_i)
    n_min= get_location(extr,2)[0].shape[0]
    n_max= get_location(extr,-2)[0].shape[0]
    return [n_min,n_max]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))

def split_series(feature_i,indexes):
    n_pieces=indexes.shape[0]
    pieces=[feature_i[:indexes[0]]]
    pieces+=[ feature_i[indexes[i-1]:indexes[i]]
                for i in range(1,n_pieces)]
    pieces.append(feature_i[indexes[-1]:])
    return pieces   

def relative_location(array_i,value):
    pos=np.where(array_i==value)[0][0]
    size=array_i.shape[0]
    return float(pos)/float(size)

def get_window(indexes,feature_i,k=5):
    def window_helper(j):
        start=0 if( j-k <0) else (j-k)
        return feature_i[start:(j+k)]
    return [ window_helper(j) for j in indexes] 

def get_location(feature_i):
    extr_i=np.abs(local_extr(feature_i))
    return np.where(extr_i==2)[0]

def relative_residuals(piece_i):
    if(piece_i.shape[0]==0):
        return [0.0]
    piece_i=piece_i.reshape( -1,1)
    pred_i=fit_linear(piece_i)
    res_i= (piece_i-pred_i)#.T[0]
    return list(res_i)#np.mean(res_i) /np.mean(piece_i)

def fit_linear(piece_i):
    x_i=np.arange(piece_i.shape[0]).reshape(-1,1)
    reg=LinearRegression()
    reg.fit(x_i, piece_i)
    return reg.predict(x_i)