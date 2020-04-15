import numpy as np
import pickle, sys
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
COLORS = ['red', 'blue']
from cvxopt import matrix as cvxmat, sparse, spmatrix
from cvxopt.solvers import qp, options
import numpy as np
def _convert(H, f, Aeq, beq, lb, ub):                                                                                  
    P = cvxmat(H)                                                                                                      
    q = cvxmat(f)
    if Aeq is None:                                                                                                    
        A = None                                                                                                       
    else: 
        A = cvxmat(Aeq)                                                                                                
    if beq is None:                                                                                                    
        b = None                                                                                                       
    else: 
        b = cvxmat(beq)                                                                                                
    
    n = lb.size
    G = sparse([-speye(n), speye(n)])                                                                                  
    h = cvxmat(np.vstack([-lb, ub]))                                                                                      
    return P, q, G, h, A, b 
#########
def speye(n):
    """Create a sparse identity matrix"""
    r = range(n)
    return spmatrix(1.0, r, r)
def read_data(f):
    with open(f, 'rb') as f:
        data = pickle.load(f)
    x, y = data[0], data[1]
    return x, y
def fit(x, y,C1,C2):
    NUM = x.shape[0]
    DIM = x.shape[1]
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((NUM, 1)))
    #G = matrix(-np.eye(NUM))
    #h = matrix(np.zeros(NUM))
    Aeq = matrix(y.reshape(1, -1))
    beq = matrix(np.zeros(1))
### framing lb and ub
### lb=np.zeros(NUM,1) ### ub=np.zeros(NUM,2)
    lb=np.zeros([NUM,1]) ### ub=np.zeros(NUM,2)
    ub=np.zeros([NUM,1])
    cc= y[:,None]
    for index in range(NUM):
	label=cc[index][0]
	if(label == -1.0):
		ub[index][0]=C1
	else:
		ub[index][0]=C2
    P, q, G, h, A, b = _convert(K, q, Aeq, beq, lb, ub)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    obj=sol['dual objective']
    return alphas,obj
############## Finding the weighted average of the error
import h5py
import matplotlib.pyplot as plt
import numpy as np
f = h5py.File('toy.hdf5','r')
x=np.transpose(f['X'][:])
y=f['y'][:]
c1=1
c2=1
classes=[-1,1]
loss=0
alphas,obj=fit(x, y,c1,c2)
w = np.sum(alphas * y[:, None] * x, axis = 0)    
cond = (alphas > 1e-4).reshape(-1)
b = y[cond] - np.dot(x[cond], w)
bias = b[0]
norm = np.linalg.norm(w)
w, bias = w / norm, bias / norm
num=x.shape[0]
for i in range(num):
	dist=np.dot(x[i,:],w)+bias
        if((dist<0) and (y[i]==1)) :
		loss=loss+c1
	elif((dist>0) and (y[i] ==-1.0)):
		loss=loss+c2
	else :
		loss=loss+0
###### weighted error loss
print 'weighted loss'
print loss
