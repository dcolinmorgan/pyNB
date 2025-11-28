import numpy as np
from scipy.sparse import random as sprand

def scalefree(N, n, *varargin):
    """
    Create a scalefree network with N nodes and specific sparseness with preferential attachment

    Parameters:
    N: number of nodes
    n: wished average number of links per node if n>=1, else relative sparsity of total possible links.
    varargin: pin, pout, seed, rank_check (logical)

    Returns:
    A: undirected scalefree network matrix
    """
    rank_check = True
    pin = np.nan
    pout = np.nan
    seed = None

    # Parse varargin
    for arg in varargin:
        if isinstance(arg, (int, float)) and np.isnan(pin):
            pin = arg
        elif isinstance(arg, (int, float)) and np.isnan(pout):
            pout = arg
        elif isinstance(arg, bool):
            rank_check = arg
        elif isinstance(arg, np.ndarray):
            seed = arg

    # Default values
    if np.isnan(pin):
        pin = 0.5
    if np.isnan(pout):
        pout = 0.3

    # Set sparsity
    if n < 1:
        sparsity = n
    else:
        sparsity = n / N

    m0 = round(sparsity * N)

    if seed is None:
        # Create seed
        seed_size = m0 * 2
        density = m0 / (m0 ** 2) if m0 > 0 else 0
        seed_sparse = sprand(seed_size, seed_size - 1, density=density, data_rvs=np.random.randn)
        seed = np.abs(seed_sparse.toarray()) > 0  # logical

        k = 0
        if rank_check:
            while np.linalg.matrix_rank(seed.astype(float)) < min(seed.shape):
                seed_sparse = sprand(seed_size, seed_size - 1, density=density, data_rvs=np.random.randn)
                seed = np.abs(seed_sparse.toarray()) > 0
                seed.flat[np.random.randint(seed.size)] = True
                k += 1
                if k % 100 == 0:
                    print(f'k = {k}')

        tmp = np.zeros((seed_size, seed_size))
        for i in range(seed_size):
            tmp[i, i+1:] = seed[i, :seed_size-1-i]
        for i in range(1, seed_size):
            tmp[i, :i] = seed[i, :i]
        seed = tmp.astype(bool)

    # Check for zero rows
    zero_rows = np.where(np.all(seed == 0, axis=1))[0]
    for zr in zero_rows:
        seed[zr, np.random.randint(seed.shape[1])] = True

    # Check for zero columns
    zero_cols = np.where(np.all(seed == 0, axis=0))[0]
    for zc in zero_cols:
        seed[np.random.randint(seed.shape[0]), zc] = True

    # Initialize A
    A = np.zeros((N, N))
    seed_rows, seed_cols = seed.shape
    A[:seed_rows, :seed_cols] = seed.astype(float)

    # Add more links
    for i in range(m0 * 2, N):
        m = m0
        if m == 0:
            m = 1

        k = 0
        while k < m:
            r = np.random.rand()
            ps = 0
            for inode in range(i):
                pl = np.sum(np.abs(A[inode, :i])) / np.count_nonzero(A[:i, :i])
                ps += pl
                if r < ps:
                    r2 = np.random.rand()
                    randC = np.random.rand()
                    val = -1 if randC <= 0.5 else 1

                    if r2 < pin:
                        A[i, inode] = val
                    elif r2 < pin + pout:
                        A[inode, i] = val

                    k += 1
                    break

    # Set diagonal to -1
    np.fill_diagonal(A, -1)

    return A
