import numpy as np
from numpy import linalg as LA

def pearson(v1, v2):
  n = len(v1)
  var_1 = np.array(v1)
  var_2 = np.array(v2)
  #Simple sums
  sum1 = np.sum(var_1)
  sum2 = np.sum(var_2)

  # Sums of the squares
  sum1Sq=np.sum(np.power(var_1, 2))
  sum2Sq=np.sum(np.power(var_2, 2))

  # Sum of the products
  pSum=np.sum(var_1*var_2)

  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/n)
  den=np.sqrt((sum1Sq-np.power(sum1,2)/n)*(sum2Sq-np.power(sum2,2)/n))
  if den==0: return 0

  return 1.0-num/den

def eucledian(raw_p, raw_q):
  p = np.array(raw_p)
  q = np.array(raw_q)

  return np.sqrt(np.sum((p-q)**2))

def manhattan(raw_p, raw_q):
  x = np.array(raw_p)
  y = np.array(raw_q)

  return np.abs(x-y).sum()
  
def eucledian2(raw_p, raw_q):
  x = np.array(raw_p)
  y = np.array(raw_q)
  dist= np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
  return dist

def cosine(raw_p, raw_q):
    a = np.array(raw_p)
    b = np.array(raw_q)
    result = np.dot(a, b)/(LA.norm(a)*LA.norm(b))
    return result