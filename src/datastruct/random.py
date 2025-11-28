import numpy as np
from scipy.sparse import random as sprandn

def randomNet(N, n):
    """
    Creates a random undirected network with N nodes and specific sparseness with no self loops.

    Parameters:
    N: number of nodes
    n: wished average number of links per node if n>=1, else relative sparsity of total possible links.

    Returns:
    A: undirected random network matrix
    """
    if n < 1:
        sparsity = n
    else:
        sparsity = n / N

    # Approximate tspar as sparsity (simplified from MATLAB calculation)
    tspar = sparsity

    A = np.zeros((N, N))

    # Generate sparse random matrix (equivalent to sprandn)
    tmp = (1 + np.random.rand() * 9) * sprandn(N, N-1, density=tspar, data_rvs=np.random.randn).toarray()

    # Fill upper triangle
    for i in range(N):
        A[i, i+1:] = tmp[i, i:]

    # Fill lower triangle to make undirected (symmetric)
    for i in range(1, N):
        A[i, :i] = A[:i, i]  # Copy upper to lower for symmetry

    return A
