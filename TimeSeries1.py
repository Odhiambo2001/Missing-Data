import numpy as np

def backward_xs_model(C, F_hat):
    # C: (T, N, K+1) array of observed characteristics
    # F_hat: (T, N, K) array of estimated contemporaneous XS factors
    
    T, N, K = F_hat.shape
    beta_hat_BXS = np.zeros((K+1, N, K+1))
    
    # Stack observed characteristics and lagged XS factors into a large vector
    X = np.zeros((N*T, K+1))
    Y = np.zeros((N*T, K+1))
    for t in range(1, T):
        X[(t-1)*N:t*N, :-1] = C[t-1, :, :-1]
        X[(t-1)*N:t*N, -K:] = F_hat[t-1, :, :]
        Y[(t-1)*N:t*N, :] = C[t, :, :]
    
    # Regress Y on lagged X to estimate beta_hat_BXS
    Xlag = X[:-N, :]
    Ylag = Y[1*N:, :]
    for i in range(N):
        beta_hat_BXS[:, i, :-1] = np.linalg.lstsq(Xlag[i*K:(i+1)*K, :], Ylag[i*K:(i+1)*K, :-1], rcond=None)[0]
    
    return beta_hat_BXS

# Example input
T, N, K = 3, 2, 2
C = np.random.rand(T, N, K+1)
F_hat = np.random.rand(T, N, K)

# Call the function and assign the output to a variable
beta_hat_BXS = backward_xs_model(C, F_hat)

# Print the output
print(beta_hat_BXS)
