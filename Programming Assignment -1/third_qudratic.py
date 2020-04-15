####### Gradient Function for Qudratic Function( with given intial values and step sizes)
import numpy as np
import matplotlib.pyplot as plt

###Set of Intial points and step size
intial_values=[[2,3],[-4,2],[0,-5.5]]
step_size_set=[0.1,0.01,0.3]
# The data to fit
N = 1000
def quadratic_function(x,y):
    function1=1.125*(x**2)+(0.5*x*y)+0.75*(y**2)+2*x+2*y
    return function1

x = np.arange(-6.0,6.0,0.1)
y = np.arange(-6.0,6.0,0.1)
X,Y = np.meshgrid(x, y)
J_grid = quadratic_function(X,Y)

# A labeled contour plot for the RHS cost function
#X, Y = np.meshgrid(theta0_grid, theta1_grid)

#ax[1].clabel(contours)
# The target parameter values indicated on the cost function contour plot
#ax.scatter([theta0_true]*2,[theta1_true]*2,s=[50,10], color=['k','w'])

# Take N steps with learning rate alpha down the gradient,
# starting at (x0, y0)

colors = ['b', 'g', 'm', 'c', 'orange']
N = 1000
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6.15))
contours = ax.contour(X, Y, J_grid, 40)
for k in range(3):
    xy_values = []
    xy_values.append(intial_values[k])
    J = [quadratic_function(*xy_values[0])]
    for j in range(1,N-1):
        last_xy = xy_values[j-1]
        this_xy = np.empty((2,))
        step_size=1
        beta=0.5
        alpha=0.5
        number_back=10
        grad_x=((2*1.125*last_xy[0])+(0.5*last_xy[1])+2)
        grad_y=((0.5*last_xy[0])+(1.5*last_xy[1])+2)
        ####Back tracking
        xx = last_xy[0]- step_size*grad_x
        yy = last_xy[1]- step_size*grad_y
        vect1=[xx,yy]
        vect2=[last_xy[0],last_xy[1]]
        while(quadratic_function(xx,yy)>(quadratic_function(last_xy[0],last_xy[1])-alpha*step_size*np.inner(vect1,vect2))):
            step_size=beta*step_size
            xx = last_xy[0]- step_size*grad_x
            yy = last_xy[1]- step_size*grad_y
            vect1=[xx,yy]
            vect2=[last_xy[0],last_xy[1]]    
        this_xy[0] = last_xy[0] - step_size*grad_x
        this_xy[1] = last_xy[1] - step_size*grad_y
        xy_values.append(this_xy)
        J.append(quadratic_function(*this_xy))
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
titlex='Contour plot for quadratic functions with step size(Back_tracking)'
ax.set_title(titlex)
plt.show()
