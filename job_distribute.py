import sys
import subprocess
import multiprocessing
import random
from run import *

q_paras = None

def worker(node):
  for kvl in iter(q_paras.get, 'STOP'):
    k, v, l, dst = kvl
    ret = subprocess.call(['ssh', node, 'source ~/.bash_profile;cd git/dcs/%s;python ../../run.py %d %s %s' %(dst, k, v, l)])

if __name__ == '__main__':
  K = range(1,20,2)
  #V = [0.1, 0.2, 0.4]
  #L = [0.05, 0.1, 0.3]
  V = [0.2]
  L = [0.3]
  kv_list = list(itertools.product(K, V))
  kl_list = list(itertools.product(K, L))
  kvl_list = list(itertools.product(K, V , L))
  #random.shuffle(kvl_list)

  nodes = ['gpu%02d' %i for i in xrange(5,12)]
  #nodes = nodes[:3][::-1] + nodes
  nodes = nodes + nodes[::-1]

  q_paras = multiprocessing.Queue()
  workers = [multiprocessing.Process(target=worker, args=(node,)) for node in nodes]
  dst_folder = sys.argv[1]
  if sys.argv[2]=='lec':
    for k,l in kl_list:
      q_paras.put((k,None,l,dst_folder))
  elif sys.argv[2]=='lmnn':
    for k,v in kv_list:
      q_paras.put((k,v,None,dst_folder))
  elif sys.argv[2]=='only':
    para_list = [(k,None,l,dst_folder) for k,l in kl_list]+[(k,v,None,dst_folder) for k,v in kv_list]
    random.shuffle(para_list)
    for para in para_list:
      q_paras.put(para)
  elif sys.argv[2]=='all':
    para_list = [(k,None,l,dst_folder) for k,l in kl_list]+[(k,v,None,dst_folder) for k,v in kv_list]+[(k,v,l,dst_folder) for k,v,l in kvl_list]
    random.shuffle(para_list)
    for para in para_list:
      q_paras.put(para)
  else:
    kvl_list = list(itertools.product(K, [0.2], [0.3]))[::-1]
    for k,v,l in kvl_list:
      q_paras.put((k,v,l,dst_folder))

  for node in nodes:
    q_paras.put('STOP')

  for p_worker in workers:
    p_worker.start()
  for p_worker in workers:
    p_worker.join()
