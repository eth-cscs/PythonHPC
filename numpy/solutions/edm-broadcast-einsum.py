def euclidean_broadcast(x, y):
    """Euclidean square distance matrix.
    
    Inputs:
    x: (N,) numpy array
    y: (N,) numpy array
    
    Ouput:
    (N, N) Euclidean square distance matrix:
    r_ij = x_ij^2 - y_ij^2
    """
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]

    return np.einsum('ijk,ijk->ij', diff, diff)
