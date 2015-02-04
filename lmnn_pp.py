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

  def process_input(self, X_train, y_train, pp_train, X_test, y_test, pp_test):
    self.X_train = X_train;self.y_train = y_train
    self.pp_train = pp_train
    self.X_val = None
    self.X_test = X_test;self.y_test = y_test
    self.pp_test = pp_test
    self.M = np.eye(X_train.shape[1])
    self.G = np.zeros(self.M.shape)
    self.active_set = None
    self.ij = []
    self.ijl = []
    self.loss = np.inf
    self.pd_pp = pairwise_distances(self.pp_train, metric='hamming')
    np.fill_diagonal(self.pd_pp, np.inf)
    if self.X_val is not None:
      #self.pp_val = pp_val
      self.pd_pp_val = pairwise_distances(self.pp_val, self.pp_train, metric='hamming')
    self.pd_pp_test = pairwise_distances(self.pp_test, self.pp_train, metric='hamming')
    self.mm = []
    self.stats= []

  def fit(self, max_iter, ff):
    # update M iteratively
    for n_iter in xrange(max_iter):
      self.n_iter_total += 1
      self.pd_X = np.square(\
      pairwise_distances(self.X_train, metric='mahalanobis', VI=self.M))
      np.fill_diagonal(self.pd_X, np.inf)
      if self.X_val is not None:
        self.pd_X_val = np.square(\
	pairwise_distances(self.X_val, self.X_train, metric='mahalanobis', VI=self.M))
      diff_G = np.zeros(self.M.shape)
      new_ijl = None
      if n_iter%10 == 0:
	#if n_iter+self.n_iter_total == 0:
	if True:
	  #initialize targets and impostors
          #new_ij, new_ijl = self._select_targets_impostors_pool(self.v, self.c)
          new_ij, new_ijl = self._select_targets_impostors()
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
	  sum_outer_product(self.X_train, income_ij[:,0], income_ij[:,1], None)
	if outcome_ij.shape[0] > 0:
	  diff_G -= (1-self.mu) *\
	  sum_outer_product(self.X_train, outcome_ij[:,0], outcome_ij[:,1], None)

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
        (sum_outer_product(self.X_train, income_ijl[:,0], income_ijl[:,1], weights)\
	- sum_outer_product(self.X_train, income_ijl[:,0], income_ijl[:,2], weights))
      if outcome_ijl.shape[0] > 0:
        weights = None
        #weights = self.pd_pp[outcome_ijl[:,0], outcome_ijl[:,2]]
        diff_G -= self.mu *\
	(sum_outer_product(self.X_train, outcome_ijl[:,0], outcome_ijl[:,1], weights)\
	- sum_outer_product(self.X_train, outcome_ijl[:,0], outcome_ijl[:,2], weights))

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

      # report status periodically
      if self.n_iter_total % 10 == 0:
	print 'fold', ff
        self.report()

      # stop if step size too small
      if self.alpha < 1e-8:
	print 'step size break'
        break

  def report(self):
    p_target_train, p_target_val, p_target_test = self.neigh_pp_mean()
    print self.n_iter_total, 'ijl: %d\t%.2f, %.4f, p_target:%.4f,%.4f'\
    %(len(self.ijl), self.loss, self.M.mean(), p_target_train[self.k/2], p_target_test[self.k/2])
    self.stats.append((len(self.ijl), self.loss, self.M.mean(), p_target_train, p_target_test))
    self.mm.append(self.M.copy())
    
  def _select_targets_impostors_pool(self, v, c):
    pool = Pool(10)
    results = pool.map(find_ijl, itertools.izip(itertools.repeat(self.k), itertools.repeat(v), itertools.repeat(c), self.pd_X, self.pd_pp))
    pool.close()
    pool.join()
    return ijl_to_list(results)

  def _select_targets_impostors(self):
    results = []
    arr_similar = self.pd_pp<self.v
    for similar, dist_x, dist_p in itertools.izip(arr_similar, self.pd_X, self.pd_pp):
      if similar.sum()>=self.k:
        similar_inds,  = np.where(similar)
	target_inds = similar_inds[dist_x[similar].argsort()[:self.k]]
      else:
        target_inds = dist_p.argsort()[:self.k]
	similar[target_inds] = True
      
      dict_jl = {}
      for ti in target_inds:
        dict_jl[ti] = np.where((dist_x < dist_x[ti] + self.c) & (~similar))[0].tolist()

      results.append(dict_jl)
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
    _, knn = NearestNeighbors(2*self.k+1, algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X_train).kneighbors(self.X_train)
    knn = knn[:,1:]
    p_target_train = np.array([(np.vstack(pd_pp[nn] for pd_pp, nn in itertools.izip(self.pd_pp, knn[:, :k]))<self.v).mean() for k in xrange(1, 2*self.k, 2)])
    #pd_pp_neigh = np.vstack(pd_pp[nn] for pd_pp, nn in itertools.izip(self.pd_pp, knn))
    #p_target_train = (pd_pp_neigh < self.v).mean()
    p_target_val = .0
    if self.X_val is not None:
      _, knn = NearestNeighbors(self.k, algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X_train).kneighbors(self.X_val)
      pd_pp_neigh = np.vstack(pd_pp[nn] for pd_pp, nn in itertools.izip(self.pd_pp_val, knn))
      p_target_val = (pd_pp_neigh < self.v).mean()
    _, knn = NearestNeighbors(2*self.k, algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X_train).kneighbors(self.X_test)
    p_target_test = np.array([(np.vstack(pd_pp[nn] for pd_pp, nn in itertools.izip(self.pd_pp_test, knn[:, :k]))<self.v).mean() for k in xrange(1, 2*self.k, 2)])
    #pd_pp_neigh = np.vstack(pd_pp[nn] for pd_pp, nn in itertools.izip(self.pd_pp_test, knn))
    #p_target_test = (pd_pp_neigh < self.v).mean()
    return p_target_train, p_target_val, p_target_test
  
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

'''
def train_lmnn(X_y_tr_te):
  X, y, (train, test), ff = X_y_tr_te
  max_iter = 500
  k = 7;mu=0.5;c=1;v=0.2
  lmnn = LMNN_PP(k=k, alpha=1e-5, mu=mu, c=c, v=v)
  clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators=100)
  lmnn.process_input(X, y, train, None, test, clf)
  lmnn.fit(max_iter, ff)
  return lmnn

def run():
  #X, y = loadPima()
  X, y = loadIonosphere()
  nfold = 10
  folds = [(tr, te) for tr, te in StratifiedKFold(y, nfold)]

  pool = Pool(nfold)
  rr = pool.map(train_lmnn,\
  itertools.izip(itertools.repeat(X), itertools.repeat(y), folds, range(nfold)),\
  chunksize=1)
  pool.close()
  pool.join()
  return rr

def des_test_lmnn(lmnn_i):
  k = 50
  lmnn, i = lmnn_i
  if i == -1:
    return des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lmnn.clf, k)
  else:
    return des_test(lmnn.X_train, lmnn.y_train, lmnn.X_test, lmnn.y_test, lmnn.clf, k, lmnn.mm[i-1 if len(lmnn.mm)>=i else -1])

def calc_des(lmnns, idx_m):
  pool = Pool(10)
  after = pool.map(des_test_lmnn, itertools.izip(lmnns, idx_m))
  pool.close()
  pool.join()
  return np.mean(after, axis=0)
  #return after

if __name__ == '__main__':
  rr = run()
  #pickle.dump(rr, open('Ionok11mu.5c1v.3max500nfold10.pickle', 'wb'))
  #pickle.dump(rr, open('Pimak11mu.5c1v.05max500.pickle', 'wb'))
  pickle.dump(rr, open('Bloodk7mu.5c1v.2.pickle', 'wb'))
'''
