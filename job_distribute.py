import sys
import subprocess
import multiprocessing
import random
from run import *

q_paras = None

def worker(node):
  for kvl in iter(q_paras.get, 'STOP'):
    k, v, l = kvl
    ret = subprocess.call(['ssh', node, 'source ~/.bash_profile;cd git/dcs;python run.py %d %.2f %.2f' %(k, v, l)])

if __name__ == '__main__':
  nodes = ['gpu%02d' %i for i in xrange(5,12)]
  nodes = nodes + nodes
  K = range(1,20,2)
  '''
  V = [0.2,0.3,0.4]
  L = [0.3,0.35,0.4]
  '''
  V = [0.2,0.3,0.4]
  L = [0.3,0.35,0.4]
  kvl_list = list(itertools.product(K, V, L))
  random.shuffle(kvl_list)
  q_paras = multiprocessing.Queue()
  workers = [multiprocessing.Process(target=worker, args=(node,)) for node in nodes]
  for k,v,l in kvl_list:
    q_paras.put((k,v,l))
  for node in nodes:
    q_paras.put('STOP')

  for p_worker in workers:
    p_worker.start()
  for p_worker in workers:
    p_worker.join()
