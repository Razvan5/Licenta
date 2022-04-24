import constants.lattice_constants as lc
import numpy as np
from sympy import Matrix

# Trapdoors for Lattices:
# Simpler, Tighter, Faster, Smaller
# Daniele Micciancio1? and Chris Peikert2??


def frameworkTrapdoor():
    n = lc.SECURITY_DIMENSION
    m1 = lc.M1
    m2 = lc.M2
    m = m1 + m2
    q = lc.PRIME
    attr_limit = lc.ALL_ATTRIBUTES_LIMIT

    H = np.identity(n=m1, dtype=int) * q

    C = np.identity(n=m1, dtype=int)
    H_prim = H - C
    # m2 - m1 * attr_limit != 0 always
    G = np.zeros(shape=(m1, m2 - m1 * attr_limit), dtype=int)
    for i in range(0, m1):
        g_column = np.zeros(shape=(m1, attr_limit), dtype=int)
        g_column[:, attr_limit - 1] = H_prim[:, i]
        for j in range(attr_limit - 2, 0, -1):
            g_column[:, j] = g_column[:, j + 1] // lc.R
        G = np.concatenate((g_column, G), axis=1)

    P = np.zeros(shape=(m2, m1), dtype=int)
    for _m1 in range(0, m1):
        P[((_m1 + 1) * attr_limit) - 1][_m1] = 1
    P = np.flip(P, axis=1)

    T = np.identity(n=attr_limit, dtype=int)
    triangle_index = np.triu_indices(attr_limit, 1)
    T[triangle_index] = -lc.R
    U = np.kron(np.eye(m1, dtype=int), T)
    zero1 = np.zeros(shape=(m1 * attr_limit, m2 - m1 * attr_limit), dtype=int)
    zero2 = np.zeros(shape=(m2 - m1 * attr_limit, m1 * attr_limit), dtype=int)
    miniI = np.identity(n=m2 - m1 * attr_limit, dtype=int)
    U = np.asarray(np.bmat([[U, zero1],
                            [zero2, miniI]]))
    d = int(np.floor((1 + lc.SIGMA) * n * np.log10(q)))
    r_choices = np.array([0, 1, -1])
    R = np.random.choice(r_choices, size=(d, m2), p=[0.5, 0.25, 0.25], replace=True)
    R_zero = np.zeros(shape=(m1 - d, m2), dtype=int)
    R = np.concatenate((R, R_zero), axis=0)

    print(f"Used Matrices:")
    print(f"G:{G.shape}\n {G}")
    print(f"R:{R.shape}\n {R}")
    print(f"U:{U.shape}\n {U}")
    print(f"P:{P.shape}\n {P}")
    print(f"C:{C.shape}\n {C}")
    print(f"Checks: G @ P \n {G @ P} \n == \n{H_prim} (H')")
    print(f"Checks: G @ P - H' : {np.count_nonzero(G @ P - H_prim)} nonzero values")

    return G, R, U, P, C


def generateTrapdoor():
    n = lc.SECURITY_DIMENSION
    m1 = lc.M1
    m2 = lc.M2
    m = m1 + m2
    q = lc.PRIME

    G, R, U, P, C = frameworkTrapdoor()
    A1 = np.random.randint(0, q - 1, size=(n, m1))
    A2 = ((-A1) @ (R + G))
    A = np.bmat([A1, A2])

    S_top = np.concatenate(((G + R) @ U, R @ P - C), axis=1)
    S_bottom = np.concatenate((U, P), axis=1)
    S = np.concatenate((S_top, S_bottom), axis=0)
    print(f"S: \n {S}")
    print(f"Checking A @ S == 0 mod q : \n {A @ S}")
    print(f"Checking A @ S == 0 mod q : \n {np.count_nonzero((A @ S) % q)} nonzero values")

    return A, S


def extendBasisRight(S_trapdoor, A_original_basis, _A_extension):
    print("ENTERING EXTEND BASE")
    prime_number = lc.PRIME
    _A_extension = (-1)*_A_extension
    print(_A_extension.T)
    W, params, _ = Matrix(A_original_basis).gauss_jordan_solve(Matrix(_A_extension), freevar=True)
    taus_zeroes = {tau: 0 for tau in params}
    W = W.xreplace(taus_zeroes)

    print("W:\n", W, W.shape)
    print("Checking A @ W:\n", A_original_basis @ W % prime_number)
    print("Checking Ā:\n", _A_extension, _A_extension.shape)
    print("Checking Ā + A @ W:\n", (_A_extension + A_original_basis @ W) % prime_number)
    SW = np.concatenate((S_trapdoor, W), axis=1)
    zeroMatrix = np.zeros(shape=(W.shape[1], W.shape[0]), dtype=int)
    identityMatrix = np.identity(n=W.shape[1], dtype=int)
    OI = np.concatenate((zeroMatrix, identityMatrix), axis=1)
    SW_OI = np.concatenate((SW, OI), axis=0)
    print("SW_OI\n", SW_OI)
    for i, line in enumerate(SW_OI):
        for j, element in enumerate(line):
            if type(element) is not int:
                print(element)
                numerator, denominator = element.as_numer_denom()
                SW_OI[i, j] = (numerator*pow(denominator, -1, lc.PRIME))
    return SW_OI % lc.PRIME
