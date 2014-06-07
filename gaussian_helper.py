import numpy as np
import pylab as pl

def generate_gaussian_data(dims, means, n_points, visualize=False):
    """ 
    dims: number of dimensions
    means: array of means for all gaussians.
    n_points: array with number of points for each gaussian.
    """

    np.random.seed(1)

    n_components = np.shape(means)[0]
    data = means[0] + np.random.randn(n_points[0], dims)
    for n in range(1, n_components):
        data = np.vstack((data, means[n] + np.random.randn(n_points[n], dims))) 


    # If visualization is set, then visualize the first 2 dimensions.    
    if visualize:
        mins = np.min(data, axis=0) - 10
        maxs = np.max(data, axis=0) + 20

        fig = pl.figure()
        ax = fig.add_subplot(111, aspect='equal')
        pl.scatter(data[:, 0], data[:, 1])
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[0])
        pl.text(mins[0] + 5, maxs[0] - 5, "Components: %d" % n_components)
        pl.text(mins[0] + 5, maxs[0] - 8, "Means: %s" % ', '.join(map(str, list(means))))
        pl.text(mins[0] + 5, maxs[0] - 11, "Points: %s" % ', '.join(map(str, list(n_points))))
        pl.show()

def main():
    means = np.array([0, 50, 60])
    n_points = np.array([100, 40, 30])
    generate_gaussian_data(2, means, n_points, True)

main()
