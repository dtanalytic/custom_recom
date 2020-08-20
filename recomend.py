import numpy as np
from scipy import sparse
from os import path

if __name__ == '__main__':
    data = np.loadtxt(r'c:\work\dev\python\progs\lessons\data\ml-100k\u.data')
    ij = data[:, :2].astype(np.int64)
    ij -= 1  # original data is in 1-based system
    values = data[:, 2]
    reviews = sparse.csc_matrix((values, ij.T)).astype(float)
    dsdsds
    # reviews = reviews.toarray()  

