import numpy as np
from numpy.linalg import eig

def get_characteristic_covariance_matrix(C):
    # calculate the characteristic covariance matrix
    T, _, L = C.shape # number of time periods and characteristics
    Nt = C.shape[1] # number of stocks at time t
    K = min(L, Nt) # number of factors
    characteristic_covariance_matrix = np.zeros((T, Nt, Nt))
    for t in range(T):
        for i in range(Nt):
            for j in range(i, Nt):
                # calculate the intersection set of observed characteristics
                Q = np.where(np.logical_and(~np.isnan(C[t, i, :]), ~np.isnan(C[t, j, :])))
                if len(Q[0]) > 0:
                    # calculate the characteristic covariance between stocks i and j at time t
                    characteristic_covariance_matrix[t, i, j] = np.sum(C[t, i, Q] * C[t, j, Q])
                    characteristic_covariance_matrix[t, j, i] = characteristic_covariance_matrix[t, i, j]
    # calculate the K largest eigenvalues and eigenvectors of the characteristic covariance matrix for each time period
    eigenvalues, eigenvectors = np.linalg.eigh(characteristic_covariance_matrix)
    idx = np.argsort(eigenvalues)[:, ::-1]
    eigenvalues = eigenvalues[np.arange(T)[:, None], idx]
    eigenvectors = eigenvectors[np.arange(T)[:, None], :, idx]
    F = eigenvectors[:, :, :K]
    return F

def get_loadings(F, C, W):
    # calculate the loadings matrix
    T, Nt, L = C.shape # number of time periods, stocks at time t, and characteristics
    K = F.shape[2] # number of factors
    Lambda = np.zeros((T, L, K))
    for t in range(T):
        for l in range(L):
            # calculate the weights matrix for characteristic l at time t
            Wl = np.diag(W[t, :, l])
            # calculate the loadings for characteristic l at time t
            Lambda[t, l, :] = np.linalg.inv(F[t, :, :].T @ Wl @ F[t, :, :]) @ (F[t, :, :].T @ Wl @ C[t, :, l])
    return Lambda

# create some sample data
c = np.random.randn(100, 10, 5)
c[0, 0, 0] = np.nan # add a missing value

def get_time_series_factor_model(C, W=None):
    # calculate the time series factor model
    if W is None:
        W = ~np.isnan(C)
    F = get_characteristic_covariance_matrix(C)
    Lambda = get_loadings(F, C, W)
    return F, Lambda

# get factor matrix and factor loadings
F, Lambda = get_time_series_factor_model(C=c, W=~np.isnan(c))

# print results
print("Factor loadings:")
print(Lambda)
print("Factor matrix")
print(F)
