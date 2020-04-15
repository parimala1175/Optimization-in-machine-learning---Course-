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
### Backtracking
colors = ['b', 'g', 'm', 'c', 'orange']
N = 1000
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6.15))
contours = ax.contour(X, Y, J_grid, 40)
for k in range(3):
    xy_values = []
    xy_values.append(intial_values[k])
    J = [Logistic_regression(*xy_values[0])]
    for j in range(1,N-1):
        last_xy = xy_values[j-1]
        this_xy = np.empty((2,))
        step_size=1
        beta=0.5
        alpha=0.5
        number_back=1
        grad_x=((2*1.125*last_xy[0])+(0.5*last_xy[1])+2)
        grad_y=((0.5*last_xy[0])+(1.5*last_xy[1])+2)
        ####Back tracking
        xx = last_xy[0]- step_size*grad_x
        yy = last_xy[1]- step_size*grad_y
        vect1=[xx,yy]
        vect2=[last_xy[0],last_xy[1]]
        while((Logistic_regression(xx,yy)>(Logistic_regression(last_xy[0],last_xy[1])-alpha*step_size*np.inner(vect1,vect2))) or (number_back == 10)):
            step_size=beta*step_size
            xx = last_xy[0]- step_size*grad_x
            yy = last_xy[1]- step_size*grad_y
            vect1=[xx,yy]
            vect2=[last_xy[0],last_xy[1]]
            number_back=number_back+1
        this_xy[0] = last_xy[0] - step_size*grad_x
        this_xy[1] = last_xy[1] - step_size*grad_y
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
titlex='Contour plot for logistic functions with step size(Back_tracking)'
ax.set_title(titlex)
plt.show()
