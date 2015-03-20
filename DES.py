import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from KNORA import KNORA

class DES_BASE:
  def __init__(self, lmnn, lec):
    self.X_train = lmnn.X_train
    self.y_train = lmnn.y_train
    self.X_test = lmnn.X_test
    self.y_test = lmnn.y_test
    self.k = lec.k
    self.update_clfs_M(lec.clf, lmnn.M)

  def update_clfs_M(self, clfs, M):
    self.clfs = clfs
    self.M = M

    '''
    _, knn_train =  NearestNeighbors(max_k+1,  algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X_train).kneighbors(self.X_train)
    self.knn_train = knn_train[:,1:]
    self.knn_test_bool =  np.array(NearestNeighbors(max_k,  algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X_train).kneighbors_graph(self.X_test).toarray(), dtype=bool)
    '''
    self.knn_test_dist, self.knn_test =  NearestNeighbors(self.k,  algorithm='brute', metric='mahalanobis', VI=self.M).fit(self.X_train).kneighbors(self.X_test)
    self.preds_train = np.array([e.predict(self.X_train) for e in clfs]).T
    self.preds_proba_train = np.array([e.predict_proba(self.X_train) for e in clfs]).swapaxes(0,1)
    self.preds_test = np.array([e.predict(self.X_test) for e in clfs]).T
    self.preds_proba_test = np.array([e.predict_proba(self.X_test) for e in clfs]).swapaxes(0,1)
    self.pp_train = np.array([pt==yt for pt,yt in itertools.izip(self.preds_train, self.y_train)])
    self.pp_test = np.array([pt==yt for pt,yt in itertools.izip(self.preds_test, self.y_test)])
    self.pd_pp_test = pairwise_distances(self.pp_test, self.pp_train, metric='hamming')

  def dcs_mcb(self, v):
    knn_test_bool = np.zeros((self.y_test.shape[0], self.y_train.shape[0]), dtype=bool)
    for ktb, nn in itertools.izip(knn_test_bool, self.knn_test):
      ktb[nn] = True
    mcb_similar_neigh = np.logical_and(knn_test_bool, self.pd_pp_test <= v)
    empty_neigh_idx = np.where(~mcb_similar_neigh.any(axis=1))[0]
    #print mcb_similar_neigh.sum(), empty_neigh_idx.shape[0]
    for i in empty_neigh_idx:
      mcb_similar_neigh[i] = knn_test_bool[i]
    return self.local_accuracy(mcb_similar_neigh)

  def dcs_ola(self):
    return self.local_accuracy(self.knn_test)

  def dcs_cla(self):
    return self.class_local_accuracy(self.knn_test)

  def dcs_prior(self):
    prior = np.array([[preds_proba[:, y] for preds_proba, y in itertools.izip(self.preds_proba_train[knn], self.y_train[knn])] for knn in self.knn_test])
    return np.vstack([d.dot(p)/d.sum() for d, p in itertools.izip(self.knn_test_dist, prior)])

  def dcs_posterior(self):
    posterior = []
    for knn, y_pred in itertools.izip(self.knn_test, self.preds_test):
      aa = np.array([self.preds_proba_train[knn][:, i, y_pred[i]] for i in xrange(self.clfs.n_estimators)])
      bb = np.array([self.y_train[knn] == yy for yy in y_pred])
      posterior.append(np.array([aaa[bbb].sum()/aaa.sum() for aaa, bbb in itertools.izip(aa, bb)]))
    return np.array(posterior)

  def knora(self):
    Knora = KNORA(self.preds_train, self.y_train, self.knn_test)
    return Knora.knora_eliminate(), Knora.knora_union()

  def select_ensemble(self, ranking, pct=None, topn=None):
    if pct is not None:
      ensembles = np.greater_equal(ranking, np.minimum(np.percentile(ranking, pct, axis=1), ranking.max(axis=1)).reshape((self.y_test.shape[0],1)))
      return ensembles
    else:
      ensembles = np.empty(ranking.shape, dtype=np.bool)
      ensembles.fill(False)
      tops = np.argsort(ranking)[:,-topn:]
      rows = np.hstack([[i]*tops.shape[1] for i in xrange(tops.shape[0])])
      cols = tops.flatten()
      ensembles[rows, cols] = True
      return ensembles

  def predict(self, ensembles, weights = None):
    proba = np.array([preds[ensemble].mean(axis=0) for ensemble, preds in itertools.izip(ensembles, self.preds_proba_test)])
    return self.clfs.classes_.take(np.argmax(proba, axis=1), axis=0)

  def local_accuracy(self, regions):
    return np.vstack([self.pp_train[region].mean(axis=0) for region in regions])

  def class_local_accuracy(self, regions):
    return np.array([self.neighorbood_precision(np.array([self.y_train[region]==c for c in self.clfs.classes_]), self.preds_train[region].T, self.preds_test[i]) for i, region in enumerate(regions)])

  def neighorbood_precision(self, u, preds_cr, pred_test):
    v = np.vstack(pc==pt for pc, pt in itertools.izip(preds_cr, pred_test))
    a = v.sum(axis=1).astype(float, copy=False)
    a[a==0] = np.inf
    b = (u[pred_test] & v).sum(axis=1)
    return np.divide(b, a)

  def pred_all(self):
    preds = {}
    rankings = {'ola': self.dcs_ola(),\
                'cla': self.dcs_cla(),\
                'prior': self.dcs_prior(),\
                'posterior': self.dcs_posterior(),\
		'mcb0': self.dcs_mcb(0),
		'mcb1': self.dcs_mcb(.05),
		'mcb2': self.dcs_mcb(.10),
		'mcb3': self.dcs_mcb(.15),
		'mcb4': self.dcs_mcb(.20),
		'mcb5': self.dcs_mcb(.25),
		'mcb6': self.dcs_mcb(.30),
		'mcb7': self.dcs_mcb(.35),
		'mcb8': self.dcs_mcb(.40),
		'mcb9': self.dcs_mcb(.45),
		}
    for key, ranking in rankings.iteritems():
      preds[key] = np.array([self.predict(self.select_ensemble(ranking, topn=n)) for n in xrange(1, self.clfs.n_estimators)])

    ensemble_kne, ensemble_knu = self.knora()
    preds['kne'] = np.tile(self.predict(ensemble_kne), (self.clfs.n_estimators-1, 1))
    preds['knu'] = np.tile(self.predict(ensemble_knu), (self.clfs.n_estimators-1, 1))
    return preds

