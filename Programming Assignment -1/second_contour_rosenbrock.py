####### Gradient Function for himmenblaus Function( with given intial values and step sizes)
import numpy as np
import matplotlib.pyplot as plt

###Set of Intial points and step size
intial_values=[[2,3],[-4,2],[2,-5.5]]
step_size_set=[0.00001,0.000002,0.000003,0.3,0.1,0.01]
# The data to fit
N = 1000
def rosenbrock(x,y):
    function_rosen=0.002*((1-x)**2)+0.2*(y*(x**2))**2
    return function_rosen
x = np.arange(-6.0,6.0,0.1)
y = np.arange(-6.0,6.0,0.1)
X,Y = np.meshgrid(x, y)
J_grid = rosenbrock(X,Y)

colors = ['b', 'g', 'm', 'c', 'orange']
N = 1000
for kx in range(3):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6.15))
    contours = ax.contour(X, Y, J_grid, 40)
    step_size=step_size_set[kx]
    for k in range(3):   
        xy_values = []
        xy_values.append(intial_values[k])
        J = [rosenbrock(*xy_values[0])]
        for j in range(1,N-1):
            last_xy = xy_values[j-1]
            this_xy = np.empty((2,))
            this_xy[0] = last_xy[0] - step_size*(-0.004*(1-last_xy[0])+0.8*last_xy[1]*last_xy[0]**2)
            this_xy[1] = last_xy[1] - step_size*(0.4*last_xy[0]**2)
            xy_values.append(this_xy)
            J.append(rosenbrock(*this_xy))
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
    titlex='Contour plot for Rosenbrock functions with step size : '+str(step_size)
    ax.set_title(titlex)
plt.show()
