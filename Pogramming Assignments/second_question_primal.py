### this program is to find the primal solution for Cost sensitive SVM using Quadratic programming
### this finds the hyper plane and decision boundary for each pair of the C1 and C2
### How decision boundary varies according the C1 and C2.
### And finding out the miss classified points after solving
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
########## hyper plane and decision boundary 
##### a function to highlight the misclassfied points(green)
## 
def decision_misclassfied(samples,labels,c1,c2):
	fig, ax = plt.subplots()
	plot_data_with_labels(samples, labels, ax)
	P,q,h,G,x,objective = cost_svm_primal_solving(samples, labels, c1,c2)
	slack= np.array(x[2:-1])
	line_params = list(x[:2]) + [x[-1]]
	xlim = pylab.gca().get_xlim()
	ylim = pylab.gca().get_ylim()
	plot_line(line_params, xlim, ylim)
	plt.hold(True)
	######plotting the decision boundary and also misclassified points
	#print slack.shape
	for i in range(100):
		if(slack[i]>1): #### wherever epsilon is greater than 1 I am treating the missclassfied points
			ax.scatter(samples[i, 0], samples[i, 1], c = 'green')
	###plotting decision boundary
	line_params = np.array(line_params)
	unique = np.unique(labels)
	slack_class1 = slack[labels == unique[0]]
	slack_class2 = slack[labels == unique[1]]
	slack_class1_ar=np.array(slack_class1)
	slack_class2_ar=np.array(slack_class2)
	slack_m_class1=min(slack_class1_ar)
	slack_m_class2=min(slack_class2_ar)
	x_sub = samples[labels == unique[0]]
	x_sub2 = samples[labels == unique[1]]
	line_params_1=np.zeros(3)
	line_params_2=np.zeros(3)
	line_params_1[0]=line_params[0]
	line_params_1[1]=line_params[1]
	line_params_1[2]=line_params[2]+1-slack_m_class1
	line_params_2[0]=line_params[0]
	line_params_2[1]=line_params[1]
	line_params_2[2]=line_params[2]-(1-slack_m_class2)
	plot_line(line_params_1, xlim, ylim)
	plot_line(line_params_2, xlim, ylim)
        plt.title('C1 and C2 values:'+str(c1)+','+str(c2))	
	pylab.show()
        
	plt.show()
        return 0;
############# this ends up the function
import h5py
import matplotlib.pyplot as plt
import numpy as np
f = h5py.File('toy.hdf5','r')
samples=np.transpose(f['X'][:])
labels=f['y'][:]
c1=1
c2=1
decision_misclassfied(samples,labels,c1,c2)
c1=10
c2=1
decision_misclassfied(samples,labels,c1,c2)
c1=1
c2=10
decision_misclassfied(samples,labels,c1,c2)
