import numpy as np
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool

class LMNN_PP:
  def __init__(self, **kwargs):
    self.params = kwargs

  def process_input(X, performance_profile):
    self.X = X
    self.performance_profile = performance_profile
    self.L = np.eye(X.shape[1])
    self.M = np.eye(X.shape[1])
    self.outer_product = {}
    self.k = self.params['k']
    self.mu = self.params['regularization']
    self.alpha= self.params['learn_rate']
    self.convergence_tol = self.params['convergence_tol']
    self.pd_pp = pairwise_distances(self.perf_profile, metric='hamming')

  def fit(self, v, c):
    ij, ijl = self._select_targets_impostors(self, v, c)
    new_ij = set(ij)-set(self.outer_product.keys())
    #for ii, jj in new_ij:
    
  def transform(self):
    return self.L.dot(self.X)

  def _select_targets_impostors(self, v, c):
    k = self.params['k']
    pd_X = pairwise_distances(self.transform())
    np.fill_diagonal(pd_X, np.inf)

    pool = Pool(10)
    results = pool.map(find_ijl, itertools.izip(itertools.repeat(k), itertools.repeat(v), itertools.repeat(c), pd_X, self.pd_pp))
    pool.close()
    pool.join()

    ij, ijl = ijl_to_list(results)
    return np.array(ij), np.array(ijl)
 
 #def evaluate_loss_function(self):


def ijl_to_list(results):
  ij = []
  ijl = []
  for i, dict_jl in enumerate(results):
    for j, ll in dict_jl.iteritems():
      ij.append((i, j))
      for l in ll:
	ijl.append((i, j, l))
  return ij, ijl

def find_ijl(kvc_xp):
  k, v, c, dist_x, dist_p = kvc_xp
  similar = dist_p < v
  if similar.sum()>=k:
    similar_inds,  = np.where(similar)
    target_inds = similar_inds[dist_x[similar].argsort()[:k]]
  else:
    target_inds = dist_p.argsort()[:k]
    similar[target_inds] = True
    
  dist_xx = dist_x * dist_x
  dict_jl = {}
  for ti in target_inds:
    dict_jl[ti] = np.where((dist_xx < dist_xx[ti] + c) & (~similar))[0].tolist()
  return dict_jl
'''
def find_ij(kvc_xp):
  k, v, c, dist_x, dist_p = kvc_xp
  similar = dist_p < v
  if similar.sum()>=k:
    similar_inds,  = np.where(similar)
    target_inds = similar_inds[dist_x[similar].argsort()[:k]]
    return target_inds, similar
  else:
    target_inds = dist_p.argsort()[:k]
    similar[target_inds] = True
    return target_inds, similar
''' 
