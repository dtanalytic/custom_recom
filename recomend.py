import numpy as np
from scipy import sparse
from os import path
import random

if __name__ == '__main__':
    data = np.loadtxt(r'../data/ml-100k/u.data')
    ij = data[:, :2].astype(int)
    ij -= 1  # original data is in 1-based system
    values = data[:, 2]
    reviews = sparse.csc_matrix((values, ij.T)).astype(float)
    
    reviews = reviews.toarray()  
    U,M = np.where(reviews)
    
    # U,M  = reviews.nonzero()
    r = random.Random(3)
    test_idxs = np.array(r.sample(range(len(U)), len(U)//10))
    train = reviews.copy()
    train[U[test_idxs], M[test_idxs]] = 0

    test = np.zeros_like(reviews)
    test[U[test_idxs], M[test_idxs]] = reviews[U[test_idxs], M[test_idxs]]
