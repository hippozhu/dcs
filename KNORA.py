import numpy as np

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

class DCSLA:
  def __init__(self, preds_cr, preds_proba_cr, y_cr, preds_test, cr):
    self.preds_cr = preds_cr
    self.preds_proba = preds_proba
    self.preds_test = preds_test
    self.y_cr = y_cr 
    self.cr = cr
    self.oracles = np.vstack(pred==yy for pred, yy in zip(self.preds_cr, self.y_cr))

  def local_accuracy(self):
    la = np.vstack([self.oracles[region].mean(axis=0) for region in self.cr])
    return np.vstack(l==m for l, m in itertools.izip(la, la.max(axis=1)))

  def class_local_accuracy(self):
    qq = np.array([precision_score(y_true, y_pred, average=None)[y_test] for y_true, y_pred, y_test in itertools.izip(itertools.repeat(des.y_cr[des.knn[0]]), des.preds_cr[des.knn[0]].T, des.preds_test[0])])
    rr = np.array([[precision_score(y_true, y_pred, average=None)[y_test] for y_true, y_pred, y_test in itertools.izip(itertools.repeat(des.y_cr[des.knn[i]]), des.preds_cr[des.knn[i]].T, des.preds_test[i])] for i in xrange(des.y_test.shape[0])])
    return 1

  def neighorbood_precision(self):
    return 1
