####### Gradient Function for Qudratic Function( with given intial values and step sizes)
import numpy as np
import matplotlib.pyplot as plt
import math

###Set of Intial points and step size
intial_values=[[2,3],[-2,2],[-4,5.5]]
step_size_set=[0.1,0.01,0.3]
# The data to fit
N = 1000
def Logistic_regression(x,y):
    function_ll=0.5*((x**2)+(y**2))+50*(math.log(1+math.exp(-0.5*y)))+50*(1+math.exp(0.2*x))
    return function_ll

x = np.arange(-6.0,6.0,0.1)
y = np.arange(-6.0,6.0,0.1)
X,Y = np.meshgrid(x, y)
f=X.shape[0]
f2=Y.shape[1]
J_grid=np.zeros([f,f2])

for i in range(f):
    for j in range(f2):
        J_grid[i][j] = Logistic_regression(X[i][j], Y[i][j]) # evaluation of the function on the grid
print J_grid.shape

colors = ['b', 'g', 'm', 'c', 'orange']
N = 1000
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6.15))
contours = ax.contour(X, Y, J_grid, 40)
for k in range(3):   
    xy_values = []
    xy_values.append(intial_values[k])
    J = [Logistic_regression(*xy_values[0])]
    for j in range(1,N-1):
        step_size=1/float(j)
        last_xy = xy_values[j-1]
        this_xy = np.empty((2,))
        this_xy[0] = last_xy[0] - step_size*((last_xy[0])+(10*np.exp(0.2*last_xy[0])))
        this_xy[1] = last_xy[1] - step_size*((last_xy[1])-(25/(np.exp(0.5*last_xy[1]+1))))
        xy_values.append(this_xy)
        J.append(Logistic_regression(*this_xy))
    for j in range(1,N-1):
        ax.annotate('', xy_values[j], xytext=xy_values[j-1],
                        arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                        va='center', ha='center')
            #ax[0].plot(x, hypothesis(x, *theta[j]), color=colors[j], lw=2,
                  # label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))
    ax.scatter(*zip(*xy_values),c=colors[k], s=40, lw=0)

    # Labels, titles and a legend.
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
titlex='Contour plot for logistic functions with step size (1/k) k- iteration number '
ax.set_title(titlex)
plt.show()
