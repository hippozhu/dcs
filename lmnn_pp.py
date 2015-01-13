import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
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
    self.n_iter_total = 0

  def process_input(self, X, pp_train, X_val=None, pp_val=None):
    self.X = X
    self.pp_train = pp_train
    self.M = np.eye(X.shape[1])
    self.G = np.zeros(self.M.shape)
    self.active_set = None
    self.ij = []
    self.ijl = []
    self.loss = np.inf
    self.pd_pp = pairwise_distances(self.pp_train, metric='hamming')
    np.fill_diagonal(self.pd_pp, np.inf)
    self.X_val = X_val
    if self.X_val is not None:
      self.pp_val = pp_val
      self.pd_pp_val = pairwise_distances(self.pp_val, self.pp_train, metric='hamming')

  def fit(self, max_iter):
    # update M iteratively
    for n_iter in xrange(max_iter):
      self.n_iter_total += 1
      self.pd_X = np.square(\
      pairwise_distances(self.X, metric='mahalanobis', VI=self.M))
      np.fill_diagonal(self.pd_X, np.inf)
      if self.X_val is not None:
        self.pd_X_val = np.square(\
	pairwise_distances(self.X_val, self.X, metric='mahalanobis', VI=self.M))
      diff_G = np.zeros(self.M.shape)
      new_ijl = None
      if n_iter%10 == 0:
	#if n_iter+self.n_iter_total == 0:
	if True:
	  #initialize targets and impostors
          new_ij, new_ijl = self._select_targets_impostors(self.v, self.c)
	  self.targets = np.array(new_ij)[:,1].reshape((len(new_ij)/self.k, self.k))
	else:
	  #only update impostors
          _, new_ijl = self._select_impostors(self.v, self.c)
	  new_ij = self.ij
	if len(new_ijl) == 0:
	  print 'no impostor'
	  break

        income_ij = np.array(list(set(new_ij)-set(self.ij)))
        outcome_ij = np.array(list(set(self.ij)-set(new_ij)))
        self.ij = new_ij

        # calculate gradient G w.r.t ij targets 
	if income_ij.shape[0] > 0:
	  diff_G += (1-self.mu) *\
	  sum_outer_product(self.X, income_ij[:,0], income_ij[:,1], None)
	if outcome_ij.shape[0] > 0:
	  diff_G -= (1-self.mu) *\
	  sum_outer_product(self.X, outcome_ij[:,0], outcome_ij[:,1], None)

	#self.active_set = set(new_ijl) | set(self.ijl)
	self.active_set = set(new_ijl) 
      else:
        new_ijl = [(i, j, l) for i, j, l in self.active_set\
	if self.c+self.pd_X[i,j]-self.pd_X[i,l] > 0]
    
      income_ijl = np.array(list(set(new_ijl)-set(self.ijl)))
      outcome_ijl = np.array(list(set(self.ijl)-set(new_ijl)))

      self.ijl = new_ijl

      # calculate gradient G w.r.t ijl triplets
      if income_ijl.shape[0] > 0:
        weights = None
        #weights = self.pd_pp[income_ijl[:,0], income_ijl[:,2]]
        diff_G += self.mu *\
        (sum_outer_product(self.X, income_ijl[:,0], income_ijl[:,1], weights)\
	- sum_outer_product(self.X, income_ijl[:,0], income_ijl[:,2], weights))
      if outcome_ijl.shape[0] > 0:
        weights = None
        #weights = self.pd_pp[outcome_ijl[:,0], outcome_ijl[:,2]]
        diff_G -= self.mu *\
	(sum_outer_product(self.X, outcome_ijl[:,0], outcome_ijl[:,1], weights)\
	- sum_outer_product(self.X, outcome_ijl[:,0], outcome_ijl[:,2], weights))

      # update M
      self.G += diff_G
      self.M -= self.alpha*self.G

      # projection to psd cone
      w, V = LA.eig(self.M)
      negatives = w < 0
      if negatives.any():
        w[negatives] = .0
	self.M = V.dot(np.diagflat(w)).dot(LA.inv(V))

      # evalute loss function
      loss = self.evaluate_loss_function()
      if loss < self.loss:
	self.alpha *= 1.1
      else:
	self.alpha /= 10
      self.loss = loss
      if self.n_iter_total % 10 == 0:
        print self.n_iter_total, 'ijl: %d\t%.2f, %.4f, alpha:%.4g, p_target:%.4f'\
        %(len(self.ijl), self.loss, self.M.mean(), self.alpha, self.neigh_pp_mean())
	#pprint(des_test(self.X_train, self.y_train, self.X_val, self.y_val, clf, 9, lmnn.M))

      if self.alpha < 1e-7:
	print 'step size break'
        break

  def _select_targets_impostors(self, v, c):
    pool = Pool(10)
    results = pool.map(find_ijl, itertools.izip(itertools.repeat(self.k), itertools.repeat(v), itertools.repeat(c), self.pd_X, self.pd_pp))
    pool.close()
    pool.join()
    return ijl_to_list(results)

  def _select_impostors(self, v, c):
    pool = Pool(10)
    results = pool.map(find_l, itertools.izip(self.targets, itertools.repeat(v), itertools.repeat(c), self.pd_X, self.pd_pp))
    pool.close()
    pool.join()
    return ijl_to_list(results)

    new_ijl = [(i, j, l) for i, j, l in self.active_set\
    if self.c+self.pd_X[i,j]-self.pd_X[i,l] > 0]
    
  def evaluate_loss_function(self):
    loss = .0
    i, j = np.array(self.ij).T.tolist()
    loss += (1-self.mu) * self.pd_X[i,j].sum()
    i, j, l = np.array(self.ijl).T.tolist()
    #print 'pp mean:', self.pd_pp[i,l].mean()
    loss += self.mu * ((self.c + self.pd_X[i,j] - self.pd_X[i,l])*self.pd_pp[i,l]).sum() 
    return loss

  def neigh_pp_mean(self):
    if self.X_val is None:
      _, knn = NearestNeighbors(self.k+1, algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X).kneighbors(self.X)
      pd_pp_neigh = np.vstack(pd_pp[nn] for pd_pp, nn in itertools.izip(self.pd_pp, knn[:, 1:]))
    else:
      _, knn = NearestNeighbors(self.k, algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X).kneighbors(self.X_val)
      pd_pp_neigh = np.vstack(pd_pp[nn] for pd_pp, nn in itertools.izip(self.pd_pp, knn))
    return (pd_pp_neigh < self.v).mean()
  
def sum_outer_product(X, I, J, pp_weight):
  if pp_weight is None:
    X_IJ = X[I] - X[J]
  else:
    X_IJ = (X[I] - X[J])*pp_weight[:, None]
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

def find_l(kvc_xp):
  target_inds, v, c, dist_x, dist_p = kvc_xp
  similar = dist_p < v
  dict_jl = {}
  for ti in target_inds:
    dict_jl[ti] = np.where((dist_x < dist_x[ti] + c) & (~similar))[0].tolist()
  return dict_jl

if __name__ == '__main__':
  from LEC import *
  k = 9 
  #nfold = 10;val_times = 2.0
  nfold = 5;val_times = 1.0
  mu=0.5
  c=1;v=0.4
  max_iter = 1000
  X, y = loadPima()
  #X, y = loadIonosphere()
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=100)
  before = []
  after = []
  ff = 0
  for train, val, test in train_val_test_split(y, nfold, val_times/(nfold-1)):
  #for train, test in StratifiedKFold(y, nfold):
    print 'fold', ff
    ff += 1
    train_val = np.hstack((train, val))
    '''
    X_train = X[train];y_train = y[train]
    X_val = X[val];y_val = y[val]
    X_test = X[test];y_test = y[test]
    '''
    clf.fit(X[train_val], y[train_val])
    before.append(des_test(X[train_val], y[train_val], X[test], y[test], clf, k))

    clf.fit(X[train], y[train])
    estimators = clf.estimators_
    preds_train = np.array(map(lambda e:e.predict(X_train), estimators)).T;pp_train = np.array([pt==yt for pt,yt in itertools.izip(preds_train, y_train)])
    preds_val = np.array(map(lambda e:e.predict(X_val), estimators)).T;pp_val = np.array([pt==yt for pt,yt in itertools.izip(preds_val, y_val)])
    lmnn = LMNN_PP(k=k, alpha=1e-5, mu=mu, c=c, v=v)
    lmnn.process_input(X_train, pp_train, X_val, pp_val)
    #lmnn.process_input(X_train, pp_train)
    lmnn.fit(max_iter)
    after.append(des_test(X_train, y_train, X_test, y_test, clf, k, lmnn.M))
  b = np.array(before);a = np.array(after)
  print ' '.join('{:6.2f}'.format(100*v) for v in b.mean(axis=0))
  print ' '.join('{:6.2f}'.format(100*v) for v in a.mean(axis=0))
  print ' '.join('{:6.2f}'.format(100*v) for v in (a-b).mean(axis=0))
  print 'mu=%.2f, c=%.1f, v=%.1f, n_iter=%d' %(mu, c, v, max_iter)
