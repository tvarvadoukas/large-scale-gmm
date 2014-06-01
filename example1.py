import numpy as np
from sklearn import mixture
np.random.seed(1)
g = mixture.GMM(n_components=2)
# Generate random observations with two modes centered on 0
# and 10 to use for training.
obs = np.concatenate((np.random.randn(100, 1),
                       10 + np.random.randn(300, 1)))
g.fit(obs) 

#GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#        n_components=2, n_init=1, n_iter=100, params='wmc',
#        random_state=None, thresh=0.01)
print np.round(g.weights_, 2)
print np.round(g.means_, 2)
print np.round(g.covars_, 2) 

g.predict([[0], [2], [9], [10]]) 

np.round(g.score([[0], [2], [9], [10]]), 2)

# Refit the model on new data (initial parameters remain the
# same), this time with an even split between the two modes.
g.fit(20 * [[0]] +  20 * [[10]]) 
#GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#        n_components=2, n_init=1, n_iter=100, params='wmc',
#        random_state=None, thresh=0.01)
np.round(g.weights_, 2)
