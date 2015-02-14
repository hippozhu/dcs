import numpy as np
from sklearn import datasets, preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold

dataset_name = None
dataset_path = {'Bupa':'/home/yzhu7/data/uci/bupa.data'}

def load_split_scale(data_file, delimiter=',', usecols=None, label_col=-1, skiprows=0):
  data = np.loadtxt(data_file, delimiter=delimiter, usecols=usecols, skiprows=skiprows)
  data = np.array(list(set(tuple(p) for p in data)))
  if label_col==-1:
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)
  else:
    X = data[:, 1:]
    y = data[:, 0].astype(np.int32)
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  return X_scaled, y
  
'''
def loadDataSet(datasetname):
  dataset_name = datasetname
  data = np.loadtxt(dataset_path[datasetname], delimiter=',')
  X = data[:, :-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1].astype(np.int32)
  return X_scaled, y
'''

def loadPima():
  data_file = '/home/yzhu7/data/diabetes/pima-indians-diabetes.data'
  return load_split_scale(data_file)

def loadBreastCancer():
  data_file = '/home/yzhu7/data/uci/breast-cancer/breast-cancer-wisconsin.data.nomissing'
  return load_split_scale(data_file)
  '''
  dataset_name = 'Breast Cancer'
  data = np.loadtxt('/home/yzhu7/data/uci/breast-cancer/breast-cancer-wisconsin.data.nomissing', delimiter=',')
  X = data[:, 1:-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1].astype(np.int32)
  return X_scaled, y
  '''

def loadIonosphere():
  data_file = '/home/yzhu7/data/uci/ionosphere/ionosphere.data'
  return load_split_scale(data_file)
  '''
  dataset_name = 'Ionosphere'
  data = np.loadtxt('/home/yzhu7/data/uci/ionosphere/ionosphere.data', delimiter=',')
  X = data[:, 1:-1]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = data[:, -1].astype(np.int32)
  return X_scaled, y
  '''

def loadIris():
  dataset_name = 'Iris'
  iris = datasets.load_iris()
  X = iris.data[50:]
  X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
  y = iris.target[50:].astype(np.int32)
  return X_scaled, y

def loadGlass():
  data_file = '/home/yzhu7/data/uci/glass/glass.data'
  return load_split_scale(data_file, usecols=range(1, 11))

def loadBlood():
  data_file = '/home/yzhu7/data/uci/transfusion/transfusion.data'
  return load_split_scale(data_file, skiprows=1)

def loadSonar():
  data_file = '/home/yzhu7/data/uci/sonar/sonar.all-data.converted'
  return load_split_scale(data_file)

def loadSpam():
  data_file = '/home/yzhu7/data/uci/spam/spambase.data'
  return load_split_scale(data_file)

def loadWine():
  data_file = '/home/yzhu7/data/uci/wine/wine.data'
  return load_split_scale(data_file, label_col=0)

def loadYeast():
  data_file = '/home/yzhu7/data/uci/yeast/yeast.data.converted'
  return load_split_scale(data_file, delimiter=None, usecols=range(1,10))
  
def loadSegmentation():
  data_file = '/home/yzhu7/data/uci/segmentation/segmentation.converted'
  return load_split_scale(data_file, label_col=0)
  
def loadMnist():
  mnist = fetch_mldata('MNIST original')
  X = mnist.data
  y = mnist.target
  return X, y

def train_val_test_split(y, nfold, test_size):
  skf = StratifiedKFold(y, nfold)
  for train_val, test in skf:
    sss = StratifiedShuffleSplit(y[train_val], 1, test_size=test_size, random_state=0)
    t, v = [(train, val) for train, val in sss][0]
    yield train_val[t], train_val[v], test

def pprint(arr):
  for i, row in enumerate(arr):
    print '[%2d]'%(i) + ' '.join('{:4.1f}'.format(100*v) for v in row)

def pprint_diff(arr):
  for i, row in enumerate(np.hstack((arr[:,0][:,None], arr[:,1:]- arr[0][0]))):
    print '[%2d]'%(i) + ' '.join('{:4.1f}'.format(100*v) for v in row)

def max_diff(arr):
  arr_diff = (arr[:,1:]- arr[0][0])
  max_val = arr_diff.max()
  max_arg = arr_diff.argmax()
  print '%.2f, (%d, %d)' %(100*max_val, max_arg/arr_diff.shape[1], max_arg%arr_diff.shape[1])
