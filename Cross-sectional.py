import numpy as np
from numpy.linalg import eig

def get_characteristic_covariance_matrix(C):
    # calculate the characteristic covariance matrix
    L = C.shape[1] # number of characteristics
    Nt = C.shape[0] # number of stocks at time t
    K = min(L, Nt) # number of factors
    characteristic_covariance_matrix = np.zeros((Nt, Nt))
    for i in range(Nt):
        for j in range(i, Nt):
            # calculate the intersection set of observed characteristics
            Q = np.where(np.logical_and(~np.isnan(C[i, :]), ~np.isnan(C[j, :])))
            if len(Q[0]) > 0:
                # calculate the characteristic covariance between stocks i and j
                characteristic_covariance_matrix[i, j] = np.sum(C[i, Q] * C[j, Q])
                characteristic_covariance_matrix[j, i] = characteristic_covariance_matrix[i, j]
    # calculate the K largest eigenvalues and eigenvectors of the characteristic covariance matrix
    eigenvalues, eigenvectors = eig(characteristic_covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    F = eigenvectors[:, :K]
    return F

def get_loadings(F, C, W):
    # calculate the loadings matrix
    Nt = C.shape[0] # number of stocks at time t
    L = C.shape[1] # number of characteristics
    K = F.shape[1] # number of factors
    Lambda = np.zeros((L, K))
    for l in range(L):
        # calculate the weights matrix for characteristic l
        Wl = np.diag(W[:, l])
        # calculate the loadings for characteristic l
        Lambda[l, :] = np.linalg.inv(F.T @ Wl @ F) @ (F.T @ Wl @ C[:, l])
    return Lambda

#create some sample data
c=np.random.randn(100,10)
c[0,0]=np.nan  #add a missing value

def get_cross_sectional_factor_model(C, W=None):
    # calculate the cross-sectional factor model
    if W is None:
        W = ~np.isnan(C)
    F = get_characteristic_covariance_matrix(C)
    Lambda = get_loadings(F, C, W)
    return F, Lambda

# get factor matrix and factor loadings
F, Lambda = get_cross_sectional_factor_model(c)

# print results
print("Factor loadings:")
print(Lambda)
print("Factor matrix")
print(F)
