import numpy as np
import itertools

class KNORA:
  def __init__(self, preds, y, cr):
    self.oracles = np.vstack(pred==yy for pred, yy in zip(preds, y))
    self.cr = cr

  def knora_eliminate(self):
    outliers = np.where(~np.any(self.oracles, axis=1))[0]
    if len(outliers) > 0:
      oracles_modified = self.oracles.copy()
      oracles_modified[outliers] = ~self.oracles[outliers]
      ensembles = np.array(map(self.oracles_intersect, (oracles_modified[nn] for nn in self.cr)))
    else:
      ensembles = np.array(map(self.oracles_intersect, (self.oracles[nn] for nn in self.cr)))
    return ensembles
  
  def knora_union(self):
    ensembles = np.array(map(self.oracles_union, (self.oracles[nn] for nn in self.cr)))
    return ensembles

  def oracles_intersect(self, neighborhood_oracles):
    common_oracles = np.all(neighborhood_oracles, axis=0)
    if common_oracles.any():
      return common_oracles
    common_oracles = np.ones(neighborhood_oracles.shape[1], dtype=bool)
    for i, no in enumerate(neighborhood_oracles):
      next_common_oracles = common_oracles & no
      if not np.any(next_common_oracles):
	break
      else:
	common_oracles = next_common_oracles
    return common_oracles

  def oracles_union(self, neighborhood_oracles):
    common_oracles = np.any(neighborhood_oracles, axis=0)
    if common_oracles.any():
      return common_oracles
    else:
      return np.ones(neighborhood_oracles.shape[1], dtype=bool)

def neighorbood_precision(u, preds_cr, pred_test):
  v = np.vstack(pc==pt for pc, pt in itertools.izip(preds_cr, pred_test))
  a = v.sum(axis=1).astype(float, copy=False)
  a[a==0] = np.inf
  b = (u[pred_test] & v).sum(axis=1)
  return np.divide(b, a)

def neighorbood_precision_proba(u, preds_proba_cr, pred_test):
  v = np.vstack(pc==pt for pc, pt in itertools.izip(preds_cr, pred_test))
  a = v.sum(axis=1).astype(float, copy=False)
  a[a==0] = np.inf
  b = (u[pred_test] & v).sum(axis=1)
  return np.divide(b, a)

def neighorbood_accuracy_proba(preds_proba_cr, yy_cr):
  return np.array([[pp[yy] for pp, yy in itertools.izip(p, yy_cr)] for p in preds_proba_cr]).mean(axis=1)

class DCSLA:
  def __init__(self, preds_cr, preds_proba_cr, y_cr, preds_test, cr):
    self.preds_cr = preds_cr
    self.preds_proba_cr = preds_proba_cr
    self.preds_test = preds_test
    self.y_cr = y_cr 
    self.cr = cr

  def local_accuracy(self):
    oracles = np.vstack(pred==yy for pred, yy in zip(self.preds_cr, self.y_cr))
    la = np.vstack([oracles[region].mean(axis=0) for region in self.cr])
    return la
    #return np.vstack(l==m for l, m in itertools.izip(la, la.max(axis=1)))

  def local_accuracy_proba(self):
    lap = np.array([neighorbood_accuracy_proba(self.preds_proba_cr[region].swapaxes(0,1), self.y_cr[region]) for region in self.cr])
    return lap

  def class_local_accuracy(self, classes):
    cla = np.array([neighorbood_precision(np.array([self.y_cr[region]==c for c in classes]), self.preds_cr[region].T, self.preds_test[i]) for i, region in enumerate(self.cr)])
    return cla

  def class_local_accuracy_proba(self, classes):
    cla = np.array([neighorbood_precision(np.array([self.y_cr[region]==c for c in classes]), self.preds_cr[region].T, self.preds_test[i]) for i, region in enumerate(self.cr)])
    return cla
