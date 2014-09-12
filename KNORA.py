import numpy as np

class KNORA:
  def __init__(self, oracles, knn):
    self.oracles = oracles
    self.knn = knn

  def knora_eliminate(self):
    outliers = np.where(~np.any(self.oracles, axis=1))[0]
    if len(outliers) > 0:
      oracles_modified = self.oracles.copy()
      oracles_modified[outliers] = ~self.oracles[outliers]
      ensembles = np.array(map(self.oracles_intersect, (oracles_modified[nn] for nn in self.knn)))
    else:
      ensembles = np.array(map(self.oracles_intersect, (self.oracles[nn] for nn in self.knn)))
    return ensembles
  
  def knora_union(self):
    ensembles = np.array(map(self.oracles_union, (self.oracles[nn] for nn in self.knn)))
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

class DCS_LA:
  
