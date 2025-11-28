import numpy as np
import cvxpy as cp

def stabilize(Atilde, **kwargs):
    """
    Stabilize weights a static network structure

    Parameters:
    Atilde: Network model, with specific desired structure (matrix)
    iaa: wished Interrampatteness, low/high (string) (low)
         low  => cond < N/2
         high => cond > N*2
         N = #nodes
    sign: if signed structure is to be kept (logical) (false)
          may not be solveable for some signed structures.

    Note: low iaa guarantees a GRN with low iaa, but setting high iaa
    does not guarantee a GRN with high iaa, hence this must be checked
    for externally.
    This function was designed for ~10-50 gene GRNs.
    """
    # Default options
    iaa = kwargs.get('iaa', 'low')
    sign = kwargs.get('sign', False)

    if iaa == 'low':  # always low
        Epsilon = -0.01
        Gamma = -10
    elif iaa == 'high':  # mix of high and low (no guarantee to have all high)
        Epsilon = -0.01
        Gamma = -100
        Atilde = 10 * Atilde

    # Stabilize matrix
    tol = 1e-4
    S = np.abs(Atilde) < tol
    n = Atilde.shape[0]
    I = np.eye(n)
    Psi = Gamma * 0.9
    Shi = Epsilon * 1.1

    # CVX optimization
    D = cp.Variable((n, n))
    g = cp.Variable()
    e = cp.Variable()

    constraints = [
        g * I <= (Atilde + D) + (Atilde + D).T,
        (Atilde + D) + (Atilde + D).T <= e * I,
        Gamma <= g,
        g <= Psi,
        Shi <= e,
        e <= Epsilon,
        D[S] == 0
    ]

    if sign:
        Neg = Atilde < -tol
        Pos = Atilde > tol
        Atmp = Atilde + D
        constraints.extend([
            Atmp[Pos] >= 0,
            Atmp[Neg] <= 0
        ])

    prob = cp.Problem(cp.Minimize(cp.norm(D, 'fro')), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"CVX optimization failed: {prob.status}")
            return Atilde  # Return original if failed
    except Exception as ex:
        print(f"CVX error: {ex}")
        return Atilde

    A = Atilde + D.value
    return A
