import numpy as np

def esf(Z):
    """
    Calculate elementary symmetric function using Mahler's recursive formula
    
    cardinality 1: r1 + r2 + .. + rn
    cardinality 2: r1*r2 + r1*3 + ... + r2*3 + ..

    Parameters
    ----------
    Z: array_like
        Input vector

    Returns
    -------
    out: ndarray
    """
    n_z = len(Z)
    if n_z == 0:
        return np.ones(1)

    F = np.zeros((2, n_z))
    i_n = 0
    i_n_minus = 1

    for n in range(n_z):
        F[i_n, 0] = F[i_n_minus, 0] + Z[n]
        for k in range(1, n + 1):
            if k == n:
                F[i_n, k] = Z[n] * F[i_n_minus, k - 1]
            else:
                F[i_n, k] = F[i_n_minus, k] + Z[n] * F[i_n_minus, k - 1]

        i_n, i_n_minus = i_n_minus, i_n

    return np.concatenate((np.ones(1), F[i_n_minus,:]))
