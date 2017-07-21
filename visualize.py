import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def f(x):
    return x[0]**2 + x[1]**2

def gradient_f(x):
    return np.array([2*x[0], 2*x[1]])

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Plot cost surface
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(X, Y)
Z = f((X, Y))
print(Z)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, alpha=0.9)

# Plot x0
x0 = np.array([0.5, 0.5])
y0 = f(x0)
ax.plot([x0[0]], [x0[1]], [y0], 'ro')

# Plot gradient
dy0 = gradient_f(x0)
length = 0.2
dx = dy0*length/np.linalg.norm(dy0)
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='g', shrinkA=0, shrinkB=0)
arrow1 = Arrow3D([x0[0], x0[0]+dx[0]], [x0[1], x0[1]+dx[1]], [y0, y0+np.dot(dy0, dx)], **arrow_prop_dict)
arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='b', shrinkA=0, shrinkB=0)
arrow2 = Arrow3D([x0[0], x0[0]-dx[0]], [x0[1], x0[1]-dx[1]], [y0, y0+np.dot(dy0, -dx)], **arrow_prop_dict)
ax.add_artist(arrow1)
ax.add_artist(arrow2)

# Plot equidistant points
n = 100
angle = np.linspace(0, 2*np.pi, n)
dx = np.concatenate((np.cos(angle).reshape(n, 1), np.sin(angle).reshape(n, 1)), axis=1)*length
xs = x0 + dx
y = np.array([y0 + np.dot(dy0, (x - x0)) for x in xs])
plt.plot(xs[:,0], xs[:,1], y)

# Plot tangent space
dX = np.array([-1, 1]) * 0.7
dY = np.array([-1, 1]) * 0.7
dX, dY = np.meshgrid(dX, dY)
y = np.array([y0 + np.dot(dy0, [dx, dy]) for dx, dy in zip(dX.flatten(), dY.flatten())]).reshape(2, 2)
ax.plot_surface(x0[0] + dX, x0[1] + dY, y, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, alpha=0.9)
plt.show()