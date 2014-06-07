import pylab as pl
import numpy as np
import matplotlib as mpl
from sklearn.mixture import GMM
from largegmm import largegmm

def make_ellipses(gmm, ax):
    for n, color in enumerate('rbg'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

np.random.seed(1)
g = largegmm(n_components=3, covariance_type='full')

# Generate observations.
data = np.concatenate((np.random.randn(1000, 2),
                       np.array([3,2]) + np.random.randn(2000, 2),
                       np.array([8,9]) + np.random.randn(4000, 2)))
g.stepwiseEM(data)
#g.fit2(data)
#g.fit(data) 

weights = np.round(g.weights_, 2)
means = np.round(g.means_, 2)
covars = np.round(g.covars_, 2) 
logprob = g.score(data)

print covars

fig = pl.figure()
ax = fig.add_subplot(111, aspect='equal')
make_ellipses(g, ax)
pl.scatter(data[:, 0], data[:, 1])

pl.text(-5, 18, 'Means: (%.2f %.2f), (%.2f %.2f) (%.2f %.2f)' % 
            (means[0][0], means[0][1], means[1][0], means[1][1], means[2][0], means[2][1]))

ax.set_xlim(-10, 20)
ax.set_ylim(-10, 20)
pl.show()
