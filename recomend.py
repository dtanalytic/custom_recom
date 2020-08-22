import numpy as np
from scipy import sparse
from os import path
import random

if __name__ == '__main__':
    data = np.loadtxt(r'../data/cus_rec/data')
    ij = data[:, :2].astype(int)
    ij -= 1  # original data is in 1-based system
    values = data[:, 2]
    rates = sparse.csc_matrix((values, ij.T)).astype(float)
    
    rates = rates.toarray()  
    U,M = np.where(rates)
    
    # U,M  = rates.nonzero()
    r = random.Random(3)
    test_idxs = np.array(r.sample(range(len(U)), len(U)//10))
    train = rates.copy()
    train[U[test_idxs], M[test_idxs]] = 0

    test = np.zeros_like(rates)
    test[U[test_idxs], M[test_idxs]] = rates[U[test_idxs], M[test_idxs]]
