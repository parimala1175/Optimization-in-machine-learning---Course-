import numpy as np
import pickle, sys
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
COLORS = ['red', 'blue']
from cvxopt import matrix as cvxmat, sparse, spmatrix
from cvxopt.solvers import qp, options
import numpy as np
########### Dual of the cost senstive SVM  
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
def plot_data_with_labels(x, y, ax):
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])
def plot_separator(ax, w, b):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.arange(-6, 12)
    ax.plot(x, x * slope + intercept, 'k-')
def hyper_plane_svm(x,y,c1,c2):
	alphas,obj = fit(x, y,c1,c2)
	w = np.sum(alphas * y[:, None] * x, axis = 0)    
	cond = (alphas > 1e-4).reshape(-1)
	b = y[cond] - np.dot(x[cond], w)
	bias = b[0]
	norm = np.linalg.norm(w)
	w, bias = w / norm, bias / norm
	fig, ax = plt.subplots()
	plot_separator(ax, w, bias)
	plot_data_with_labels(x, y, ax)
	plt.title('C1 and C2 values:'+str(c1)+','+str(c2))
	fig, ax2 = plt.subplots()
	plot_separator(ax2, w, bias)
	### alpha=0 (Blue) alpha=(0,Ck) (Green) alpha==Ck (Red)
        unique = np.unique(y)
        c=[c1,c2]
	for li in range(len(unique)):
        	x_sub = x[y == unique[li]]
		alphas_x=alphas[y==unique[li]]
		numx=x_sub.shape[0]
		for xi in range(numx):
			if (alphas_x[xi]<0.1):
				ax2.scatter(x_sub[xi, 0], x_sub[xi, 1], c = 'blue')
			elif((alphas_x[xi] < c[li]) and (alphas_x[xi] > 0.1)) :
				ax2.scatter(x_sub[xi, 0], x_sub[xi, 1], c = 'red')
			else:
				ax2.scatter(x_sub[xi, 0], x_sub[xi, 1], c = 'green')
        plt.title('This plot shows the location of points according to alpha value( alpha<0.1 = Blue) (Ck >alpha >0.1 = Red) (alpha>=Ck = Green)')
        ######plotting the Signed distance(primal) verses ALphas per each pair of C1 and C2 
        ###### from w and bias values we compute signed distance
        num=x.shape[0]
	signed_distance=np.zeros(num)
	for i in range(num):
                #print np.dot(x[i,:],w)*y[i]+bias
		signed_distance[i]=y[i]*(np.dot(x[i,:],w)+bias)
	plt.figure()
	plt.plot(alphas,signed_distance,'ro')
	plt.title('Plotted primal distance verses Alphas')
	plt.show()
	return obj;
###########
import h5py
import matplotlib.pyplot as plt
import numpy as np
f = h5py.File('toy.hdf5','r')
x=np.transpose(f['X'][:])
y=f['y'][:]
c1=1
c2=1
print 'first pair and objective\n'
print c1,c2
print hyper_plane_svm(x,y,c1,c2)
c1=1
c2=10
print 'second pair and objective\n'
print c1,c2
print hyper_plane_svm(x,y,c1,c2)
c1=10
c2=1
print 'Third pair and objective\n'
print c1,c2
print hyper_plane_svm(x,y,c1,c2)
############# first part of the question identifying the data point locationa based on the value of alphas
############ 
###
alphas,obj = fit(x, y,c1,c2)
