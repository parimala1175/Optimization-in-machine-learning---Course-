#### Author: Parimala
### Visualizing the fucntions
## functions defined
import math
import numpy as np
def quadratic_function(x,y):
    function1=1.125*(x**2)+(0.5*x*y)+0.75*(y**2)+2*x+2*y
    return function1
def Logistic_regression(x,y):
    function_ll=0.5*((x**2)+(y**2))+50*(math.log(1+math.exp(-0.5*y)))+50*(1+math.exp(0.2*x))
    return function_ll
def himmelblaus(x,y):
    function_h=0.1*(((x**2)+y-11)**2)+0.1*((x+(y**2)-7)**2)
    return function_h
def rosenbrock(x,y):
    function_rosen=0.002*((1-x)**2)+0.2*(y*(x**2))**2
    return function_rosen
    
######
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
# the function that I'm going to plot
x = np.arange(-6.0,6.0,0.1)
y = np.arange(-6.0,6.0,0.1)
X,Y = meshgrid(x, y) # grid of point
Z = quadratic_function(X, Y) # evaluation of the function on the grid
minx=Z.min()
maxx=Z.max()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=minx, cmap=cm.coolwarm)


ax.set_xlabel('X')
ax.set_xlim(-6, 6)
ax.set_ylabel('Y')
ax.set_ylim(-6, 6)
ax.set_zlabel('Z')
ax.set_zlim(minx, maxx)

plt.title('Quadratic_fucntion')
##### second function logistic regression
# the function that I'm going to plot
x = np.arange(-6.0,6.0,0.1)
y = np.arange(-6.0,6.0,0.1)
X,Y = meshgrid(x, y) # grid of point
f=X.shape[0]
f2=Y.shape[1]
Z=np.zeros([f,f2])

for i in range(f):
    for j in range(f2):
        Z[i][j] = Logistic_regression(X[i][j], Y[i][j]) # evaluation of the function on the grid
print Z.shape
minx=Z.min()
maxx=Z.max()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=minx, cmap=cm.coolwarm)


ax.set_xlabel('X')
ax.set_xlim(-6, 6)
ax.set_ylabel('Y')
ax.set_ylim(-6, 6)
ax.set_zlabel('Z')
ax.set_zlim(minx, maxx)
plt.title('Logistic regression function')
x = np.arange(-6.0,6.0,0.1)
y = np.arange(-6.0,6.0,0.1)
X,Y = meshgrid(x, y) # grid of point
Z = himmelblaus(X, Y) # evaluation of the function on the grid
minx=Z.min()
maxx=Z.max()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=minx, cmap=cm.coolwarm)


ax.set_xlabel('X')
ax.set_xlim(-6, 6)
ax.set_ylabel('Y')
ax.set_ylim(-6, 6)
ax.set_zlabel('Z')
ax.set_zlim(minx, maxx)
plt.title('himmelblause function')
x = np.arange(-3.0,3.0,0.1)
y = np.arange(-6.0,6.0,0.1)
X,Y = meshgrid(x, y) # grid of point
Z = rosenbrock(X, Y) # evaluation of the function on the grid
minx=Z.min()
maxx=Z.max()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=minx, cmap=cm.coolwarm)


ax.set_xlabel('X')
ax.set_xlim(-3, 3)
ax.set_ylabel('Y')
ax.set_ylim(-6, 6)
ax.set_zlabel('Z')
ax.set_zlim(minx, maxx)
plt.title('Rosenbrock function')
plt.show()
### Conclusions: ########
####
