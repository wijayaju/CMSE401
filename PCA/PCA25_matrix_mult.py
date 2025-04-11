#Attempt at a parallel multiply
import matplotlib.pylab as plt
import numpy as np
import sympy as sp
import random
import time
import multiprocessing
num_procs = multiprocessing.cpu_count()
sp.init_printing(use_unicode=True)
m = 4
d = 10
n = 4

A = [[random.random() for i in range(d)] for j in range(m)]
B = [[random.random() for i in range(n)] for j in range(d)]

def multiply(m1,m2):
    m = len(m1)
    d = len(m2)
    n = len(m2[0])
    if len(m1[0]) != d:
        print("ERROR - inner dimentions not equal")
    result = [[0 for i in range(m)] for j in range(n)]
    for i in range(0,m):
        for j in range(0,n):
            for k in range(0,d):
                result[i][j] = result[i][j] + m1[i][k] * m2[k][j]
    return result


def compute_element(args):
    i, j, m1, m2 = args
    return i, j, sum(m1[i][k] * m2[k][j] for k in range(len(m2)))

def parallel_multiply(m1, m2):
    m, d, n = len(m1), len(m2), len(m2[0])
    
    if len(m1[0]) != d:
        raise ValueError("ERROR - inner dimensions not equal")
    
    result = [[0] * n for _ in range(m)]
    
    with multiprocessing.Pool() as pool:
        indices = [(i, j, m1, m2) for i in range(m) for j in range(n)]
        for i, j, value in pool.map(compute_element, indices):
            result[i][j] = value
    
    return result
if __name__ == "__main__":
    

    #Parallel result
    start = time.time()

    parallel_answer = parallel_multiply(A, B)
    parallel_time = time.time()-start


    #Serial Result
    start = time.time()
    serial_answer = multiply(A,B)
    serial_time = time.time()-start
    
    #Numpy result
    A_ = np.matrix(A)
    B_ = np.matrix(B)

    start = time.time()

    np_answer = A_*B_
    np_time = time.time()-start

    print('np_answer =',np_time,'seconds')
    print('parallel_answer=',parallel_time,'seconds')
    print('serial_answer=',serial_time,'seconds')


    print("\n_______\nParallel\n")
    for row in parallel_answer:
        print(row)

    print("\n________\nSerial\n")
    for row in serial_answer:
        print(row)

    print("\n________\nNumpy\n")
    for row in np_answer:
        print(row)
