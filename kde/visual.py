import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

def make3dplot(ax, sample, density):
    ax.scatter(sample[:,0], sample[:,1], zdir='z')
    ax.set_aspect('equal', 'datalim')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    gridsize=50
    xs, ys = np.mgrid[xlim[0]:xlim[1]:(xlim[1]-xlim[0])/float(gridsize), ylim[0]:ylim[1]:(ylim[1]-ylim[0])/float(gridsize)]
    pos = np.empty(xs.shape + (2,))
    pos[:, :, 0] = xs; pos[:, :, 1] = ys

    zs = density(pos)

    surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
