import numpy as np


a = np.array([  [1,1,1, 2,2,2, 3,3,3],
                [1,1,1, 2,2,2, 3,3,3],
                [1,1,1, 2,2,2, 3,3,3],
                [4,4,4, 5,5,5, 6,6,6],
                [4,4,4, 5,5,5, 6,6,6],
                [4,4,4, 5,5,5, 6,6,6],
                [7,7,7, 8,8,8, 9,9,9],
                [7,7,7, 8,8,8, 9,9,9],
                [7,7,7, 8,8,8, 9,9,9]])

def refactor(a, f):
    a0 = a.reshape((a.shape[0], a.shape[1], int(a.size/(a.shape[0]*a.shape[1]))), order='F')
    a1 = a.reshape((a0.shape[0], f, int(a0.shape[1]*a0.shape[2]/f)), order='F')
    a2 = np.transpose(a1,axes=[1,0,2]).reshape((int(f**2),int(a0.size/(f**2))), order='F')
    a3 = a2.mean(axis=0).reshape((int(a0.shape[0]/f), int(a0.shape[1]/f), a0.shape[2]), order='F')
    return a3.reshape(a3.shape[0:len(a.shape)])


rf = refactor(a, 3)
print(rf[:,:])

b = np.dstack([a,a,a,a,a,a,a])
rf = refactor(b, 3)
print(rf[:,:,0])