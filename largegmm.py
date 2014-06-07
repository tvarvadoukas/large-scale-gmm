import pylab as pl
import numpy as np
from numpy.random import RandomState
import matplotlib as mpl
from sklearn.mixture import GMM
from sklearn import cluster

class largegmm(GMM):
    def __init__(self, *args, **kwargs):
        GMM.__init__(self, *args, **kwargs)
        self.log_likelihood = -235813
        self.draws = 0

    def fit2(self, X):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        ## initialization step
        X = np.asarray(X, dtype=np.float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        for _ in range(self.n_init):
            if 'm' in self.init_params or not hasattr(self, 'means_'):
                self.means_ = cluster.KMeans(
                    n_clusters=self.n_components,
                    random_state=self.random_state).fit(X).cluster_centers_

            import time
            np.random.seed(int(time.time()))
            ind = np.random.random_integers(0, X.shape[0]-1, self.n_components)
            self.means_ = X[ind, :]
            print ind

            if 'w' in self.init_params or not hasattr(self, 'weights_'):
                self.weights_ = np.tile(1.0 / self.n_components,
                                        self.n_components)

            if 'c' in self.init_params or not hasattr(self, 'covars_'):
                cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
                if not cv.shape:
                    cv.shape = (1, 1)
                self.covars_ = \
                    distribute_covar_matrix_to_match_covariance_type(
                        cv, self.covariance_type, self.n_components)

            # EM algorithms
            log_likelihood = []
            # reset self.converged_ to False
            self.converged_ = False
            for i in range(self.n_iter):

                # Expectation step
                curr_log_likelihood, responsibilities = self.score_samples(X)
                log_likelihood.append(curr_log_likelihood.sum())


                if i > 0: print log_likelihood[-1] - log_likelihood[-2]

                # Check for convergence.
                if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < \
                        self.thresh:
                    self.converged_ = True
                    break

                # Maximization step
                self._do_mstep(X, responsibilities, self.params,
                               self.min_covar)

            print i

            # if the results are better, keep it
            if self.n_iter:
                if log_likelihood[-1] > max_log_prob:
                    max_log_prob = log_likelihood[-1]
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")
        # self.n_iter == 0 occurs when using GMM within HMM
        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        return self

    def stepwiseEM(self, X, msize=100):
        """ 
        Stepwise EM. View as stochastic gradient in the space 
        of sufficient statistics.

        Input
        X: array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point. 
        msize: integer
                Size of the minibatch.
        """

        ## initialization step
        X = np.asarray(X, dtype=np.float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        for _ in range(self.n_init):
            if 'm' in self.init_params or not hasattr(self, 'means_'):
                self.means_ = cluster.KMeans(
                    n_clusters=self.n_components,
                    random_state=self.random_state).fit(X).cluster_centers_

            # Initialize means to one of the points at random.
            ind = np.random.random_integers(0, X.shape[0]-1, self.n_components)
            self.means_ = X[ind, :]
            print ind

            if 'w' in self.init_params or not hasattr(self, 'weights_'):
                self.weights_ = np.tile(1.0 / self.n_components,
                                        self.n_components)

            if 'c' in self.init_params or not hasattr(self, 'covars_'):
                cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
                if not cv.shape:
                    cv.shape = (1, 1)
                self.covars_ = \
                    distribute_covar_matrix_to_match_covariance_type(
                        cv, self.covariance_type, self.n_components)

            self.draw(X)

            # EM algorithm
            log_likelihood = []

            self.converged_ = False
            n_data = X.shape[0]

            np.random.shuffle(X)        
            for i in range(self.n_iter):
                # Shuffle the dataset. Performs in-place.
                #np.random.shuffle(X)        

                # Iterate over the mini-batches.
                iter = 0
                for ind in range(0, n_data, msize):
                    
                    # Calculate stepsize.
                    eta = 1.0 / (iter + 2)

                    # Expectation step
                    _, responsibilities = self.score_samples(X[ind:ind+msize, :])

                    # Save previous values.
                    old_weights = self.weights_
                    old_means = self.means_
                    old_covars = self.covars_

                    # Maximization step
                    self._do_mstep(X[ind:ind+msize, :], responsibilities, self.params,
                               self.min_covar)

                    # Update step. 
                    self.weights_ = (1 - eta) * old_weights + eta * self.weights_
                    self.means_ = (1 - eta) * old_means + eta * self.means_
                    self.covars_ = (1 - eta) * old_covars + eta * self.covars_

                    iter += msize

                    # TODO: remove storing the log-likelihoods.
                    curr_log_likelihood, responsibilities = self.score_samples(X)
                    self.log_likelihood = curr_log_likelihood.sum()
                    log_likelihood.append(self.log_likelihood)
                if i > 0:
                    print log_likelihood[-1] - log_likelihood[-2]

                    # Check for convergence.
                    if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < \
                            self.thresh:
                        self.converged_ = True
                        break


                if self.converged_: break

            print i                
            self.draw(X)
            pl.plot(log_likelihood)
            pl.savefig('likelihood.png', bbox_inches='tight')
            pl.close()

            # if the results are better, keep it
            if self.n_iter:
                if log_likelihood[-1] > max_log_prob:
                    max_log_prob = log_likelihood[-1]
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        return self


    def make_ellipses(self, ax):
        for n, color in enumerate('rbg'):
            v, w = np.linalg.eigh(self._get_covars()[n][:2, :2])
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v *= 9
            ell = mpl.patches.Ellipse(self.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    def draw(self, data):
        n_points = data.shape[0]
        self.draws += 1
        
        mins = np.min(data, axis=0) - 10
        maxs = np.max(data, axis=0) + 20

        pl.ion()
        fig = pl.figure()
        ax = fig.add_subplot(111, aspect='equal')
        pl.scatter(data[:, 0], data[:, 1])
        self.make_ellipses(ax)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[0])
        pl.text(mins[0] + 2, maxs[0] - 2, "Means: %s" % ', '.join(map(str, list(np.round(self.means_, 2)))))
        pl.text(mins[0] + 2, maxs[0] - 4, "Log_Likelihood: %.3f" % self.log_likelihood)
        #pl.text(mins[0] + 5, maxs[0] - 11, "Points: %s" % ', '.join(map(str, list(n_points))))
        pl.savefig('foo%d.png' % self.draws, bbox_inches='tight')        
        pl.close(fig)
        #pl.show()


def distribute_covar_matrix_to_match_covariance_type(
        tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template
    """
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv
