import numpy as np
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool
import itertools
from numpy import linalg as LA

class LMNN_PP:
  def __init__(self,
               k=3,
	       alpha=1e-5,
	       mu=0.5,
	       c=0.01,
	       v=0.3):
    self.k = k
    self.mu = mu
    self.alpha= alpha
    self.c = c
    self.v = v

  def process_input(self, X, performance_profile):
    self.X = X
    self.performance_profile = performance_profile
    self.L = np.eye(X.shape[1])
    self.M = np.eye(X.shape[1])
    self.G = np.zeros(self.M.shape)
    self.active_set = None
    self.ij = []
    self.ijl = []
    self.loss = np.inf
    #self.convergence_tol = self.params['convergence_tol']
    self.pd_pp = pairwise_distances(self.performance_profile, metric='hamming')
    np.fill_diagonal(self.pd_pp, np.inf)

  def fit(self, max_iter):
    #old_ij = self.ij
    #old_ijl = self.ijl
    #active_set = None 
    for n_iter in xrange(max_iter):
      self.pd_X = np.square(\
      pairwise_distances(self.X, metric='mahalanobis', VI=self.M))
      np.fill_diagonal(self.pd_X, np.inf)

      diff_G = np.zeros(self.M.shape)
      new_ijl = None
      if n_iter % 10 == 0:
        new_ij, new_ijl = self._select_targets_impostors(self.v, self.c)
        income_ij = np.array(list(set(new_ij)-set(self.ij)))
        outcome_ij = np.array(list(set(self.ij)-set(new_ij)))
        self.ij = new_ij

        print n_iter, 'in_ij: %d, new_ijl: %d\t%.2f, %.4f, alpha:%f'\
	%(income_ij.shape[0], len(new_ijl), self.loss, self.M.mean(), self.alpha*1e3)
	if income_ij.shape[0] > 0:
	  diff_G += (1-self.mu) *\
	  sum_outer_product(self.X, income_ij[:,0], income_ij[:,1])
	if outcome_ij.shape[0] > 0:
	  diff_G -= (1-self.mu) *\
	  sum_outer_product(self.X, outcome_ij[:,0], outcome_ij[:,1])

	#self.active_set = set(new_ijl) | set(self.ijl)
	self.active_set = set(new_ijl) 
      else:
        new_ijl = [(i, j, l) for i, j, l in self.active_set\
	if self.c+self.pd_X[i,j]-self.pd_X[i,l] > 0]
    
      income_ijl = np.array(list(set(new_ijl)-set(self.ijl)))
      outcome_ijl = np.array(list(set(self.ijl)-set(new_ijl)))

      self.ijl = new_ijl
      #print n_iter, 'income_ijl: %d, outcome_ijl: %d' %(income_ijl.shape[0], outcome_ijl.shape[0])

      if income_ijl.shape[0] > 0:
        diff_G += self.mu *\
        (sum_outer_product(self.X, income_ijl[:,0], income_ijl[:,1])\
	- sum_outer_product(self.X, income_ijl[:,0], income_ijl[:,2]))
      if outcome_ijl.shape[0] > 0:
        diff_G -= self.mu *\
	(sum_outer_product(self.X, outcome_ijl[:,0], outcome_ijl[:,1])\
	- sum_outer_product(self.X, outcome_ijl[:,0], outcome_ijl[:,2]))

      self.G += diff_G
      #print G
      self.M -= self.alpha*self.G
      w, V = LA.eig(self.M)
      negatives = w < 0
      if negatives.any():
        w[negatives] = .0
	self.M = V.dot(np.diagflat(w)).dot(LA.inv(V))
      loss = self.evaluate_loss_function()
      if loss < self.loss:
	self.alpha *= 1.1
      else:
	self.alpha /= 10
      self.loss = loss


  def transform(self):
    return self.L.dot(self.X.T).T

  def _select_targets_impostors(self, v, c):
    pool = Pool(10)
    results = pool.map(find_ijl, itertools.izip(itertools.repeat(self.k), itertools.repeat(v), itertools.repeat(c), self.pd_X, self.pd_pp))
    pool.close()
    pool.join()

    return ijl_to_list(results)
 
  def evaluate_loss_function(self):
    loss = .0
    i, j = np.array(self.ij).T.tolist()
    loss += (1-self.mu) * self.pd_X[i,j].sum()
    i, j, l = np.array(self.ijl).T.tolist()
    loss += self.mu * (self.c + self.pd_X[i,j] - self.pd_X[i,l]).sum() 
    return loss

def sum_outer_product(X, I, J):
  X_IJ = X[I] - X[J]
  return X_IJ.T.dot(X_IJ)

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
    
  dict_jl = {}
  for ti in target_inds:
    dict_jl[ti] = np.where((dist_x < dist_x[ti] + c) & (~similar))[0].tolist()
  return dict_jl

if __name__ == '__main__':
  from Dcs import *
  X, y = loadPima();folds = [(train, test) for train, test in StratifiedKFold(y, 10)];train, test = folds[0];X_train = X[train];y_train = y[train];X_test = X[test];y_test = y[test]
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100);clf.fit(X_train, y_train);estimators = clf.estimators_;preds_train = np.array(map(lambda e:e.predict(X_train), estimators)).T;preds_test = np.array(map(lambda e:e.predict(X_test), estimators)).T;perf_profile = np.array([pt==yt for pt,yt in itertools.izip(preds_train, y_train)])
  lmnn = LMNN_PP(k=5, alpha=0.0001, mu=0.5, c=0.01)
  lmnn.process_input(X_train, perf_profile)
  lmnn.fit(0.4)

