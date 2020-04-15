### this program is to find the primal solution for Cost sensitive SVM using Quadratic programming
### this finds the hyper plane and optimal solution for each pair of C1 and C2.
### Author :Parimala
import numpy as np
import pickle, sys
import numpy.random as npr
import pylab
from cvxopt import solvers, matrix
import pylab
COLORS = ['red', 'blue']
#### these three function are for plotting the hyper plane and projecting the data onto a single plane.
## scatter plots of the data points
def project(x, pt):
    i = pt[0] + -(x[0] * pt[0] + x[1] * pt[1] + x[2]) * x[0] / (x[0] ** 2 + x[1] ** 2)
    j = pt[1] + -(x[0] * pt[0] + x[1] * pt[1] + x[2]) * x[1] / (x[0] ** 2 + x[1] ** 2)
    return [i,j]
def plot_line(x, xlim, ylim):
    pt1 = [xlim[0],ylim[0]]
    pt2 = [xlim[0],ylim[1]]
    pt3 = [xlim[1],ylim[0]]
    pt4 = [xlim[1],ylim[1]]
    nt1 = project(x, pt1)
    nt2 = project(x, pt2)
    nt3 = project(x, pt3)
    nt4 = project(x, pt4)

    x = [i[0] for i in (nt1,nt2,nt3,nt4)]
    y = [i[1] for i in (nt1,nt2,nt3,nt4)]
    pylab.plot(x,y)
    pylab.gca().set_xlim(xlim)
    pylab.gca().set_ylim(ylim)
def plot_data_with_labels(x, y, ax):
    unique = np.unique(y)
    print unique
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])
#################

def cost_svm_primal_solving(samples, labels, c1,c2 ):
    n = len(samples[0])
    m = len(samples)
    nvars = n + m + 1
    ### framing the required Matrices to solve through Qudratic Programming and finding the solution
    P = matrix(0.0, (nvars, nvars))
    for i in range(n):
        P[i,i] = 1.0
    q = matrix(0.0,(nvars,1))
    ### assigning the slack variabes C1 and C2 according to the class
    ## Here I am framing my matrices such that first two(n) are my weights'(Ws)'
    ### Next M variables are my slack varibles epsilons
    ### last one is my bias value
    j=0
    for i in range(n,n+m):
        if(labels[j] == -1.0):
        	q[i] = c1
        else:
		q[i] = c2
	j=j+1
    q[-1] = 1.0
    h = matrix(-1.0,(m+m,1))
    h[m:] = 0.0
    G = matrix(0.0, (m+m,nvars))
    ### G matrix also formed according to the constraints given followign the inequalities
    ##  Here slack varriables also treated as variables so we are gettign the M+M matrices
    for i in range(m):
        G[i,:n] = -labels[i] * samples[i]
        G[i,n+i] = -1
        G[i,-1] = -labels[i]
    for i in range(m,m+m):
        G[i,n+i-m] = -1.0
    #### framed matrices are given to the qp solver
    x = solvers.qp(P,q,G,h)['x']
    primal_obl=solvers.qp(P,q,G,h)['primal objective'] ### solving using Quadratic programming tools
    return P, q, h, G, x,primal_obl
###################
########Main program Starts here Data reading and hyperplane from cost SVM primal
import h5py
import matplotlib.pyplot as plt
import numpy as np
#### LOad the data from the file
f = h5py.File('toy.hdf5','r')
samples=np.transpose(f['X'][:])
labels=f['y'][:]
fig1, ax2 = plt.subplots()
plot_data_with_labels(samples, labels, ax2)

######## plot the data for visualization for each pair of C1 and C2
fig, ax = plt.subplots()
plot_data_with_labels(samples, labels, ax)
c1=1
c2=1
P,q,h,G,x,objective = cost_svm_primal_solving(samples, labels, c1,c2)
line_params = list(x[:2]) + [x[-1]]
#print line_params
xlim = pylab.gca().get_xlim()
ylim = pylab.gca().get_ylim()
#print xlim,ylim
plot_line(line_params, xlim, ylim)
plt.title('C1 and C2 values:'+str(c1)+','+str(c2))
fig, ax = plt.subplots()
plot_data_with_labels(samples, labels, ax)
c12=1
c22=10
P,q,h,G,x2,objective2 = cost_svm_primal_solving(samples, labels, c12,c22)
line_params = list(x2[:2]) + [x2[-1]]
#print line_params
xlim = pylab.gca().get_xlim()
ylim = pylab.gca().get_ylim()
#print xlim,ylim
plot_line(line_params, xlim, ylim)
plt.title('C1 and C2 values:'+str(c12)+','+str(c22))
fig, ax = plt.subplots()
plot_data_with_labels(samples, labels, ax)
c13=10
c23=1
P,q,h,G,x3,objective3 = cost_svm_primal_solving(samples, labels, c13,c23)
line_params = list(x3[:2]) + [x3[-1]]
#print line_params
xlim = pylab.gca().get_xlim()
ylim = pylab.gca().get_ylim()
#print xlim,ylim
plot_line(line_params, xlim, ylim)
plt.title('C1 and C2 values:'+str(c13)+','+str(c23))
########## printing the objective value and and c1 and c2
print 'first pair\n'
print c1,c2, objective
print 'second \n'
print c12,c22, objective2
print 'third\n'
print c13,c23, objective3
plt.show()
pylab.show()

