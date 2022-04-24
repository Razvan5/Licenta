import numpy
from sympy import Matrix, linsolve
from random import randint
#
from constants.lattice_constants import *


# Approximate Trapdoors for Lattices and
# Smaller Hash-and-Sign Signatures
# Yilei Chen∗ Nicholas Genise† Pratyay Mukherjee‡


def gadgetSampleCut(_v, _sigma):
    pass


def approximateSamplePreimages(_a, _r, _u, _s):
    pass


def approximateTrapdoorGenerator(_lambda):
    pass


# Trapdoors for Lattices:
# Simpler, Tighter, Faster, Smaller
# Daniele Micciancio1? and Chris Peikert2??


# def trapdoorGenerator(_lambda, _n, _m, _q):
#     basis = numpy.random.randint(low=1, high=_q-1, size=(_n, _m))
#     return basis, randomVectorFromMatrix(basis)


def trapdoorGenerator(n, m, q, _a_random_matrix=None, h_invertible_matrix=None):
    print("n:", n)
    print("m:", m)
    w = numpy.random.randint(2, m - 1)
    print("w:", w)
    _m = m - w
    print("_")
    print("m:", _m)
    if _a_random_matrix is None:
        _a_random_matrix = numpy.random.randint(low=0, high=q - 1, size=(n, _m))
        print("_A:\n", _a_random_matrix)
    if h_invertible_matrix is None:
        # trisant has to be something different
        h_invertible_matrix = numpy.identity(n, dtype=int)
        h_invertible_matrix = numpy.random.randint(low=0, high=q - 1, size=(n, n))
        # needs invertible check! (if det != 0)
        print("INVERTIBLE MATRIX:\n", h_invertible_matrix)

    rng = numpy.random.default_rng()
    r_trapdoor = rng.integers(low=0, high=q - 1, size=(_m, w), endpoint=True)
    print("TRAPDOOR:\n", r_trapdoor)
    g_primitive_matrix = rng.integers(low=1, high=q - 1, size=(n, w))
    print("PRIMITIVE G:\n", g_primitive_matrix)
    a1 = h_invertible_matrix @ g_primitive_matrix - _a_random_matrix @ r_trapdoor
    a1 = a1 % q

    a = numpy.concatenate((_a_random_matrix, a1), axis=1)
    print("COMPLETE A:", a)
    print("Size a :", len(a), " ", len(a[0]))
    print("Size r :", len(r_trapdoor), " ", len(r_trapdoor[0]))

    r_i = numpy.concatenate((r_trapdoor, numpy.identity(w, dtype=int)), axis=0)
    print("[R]")
    print("[I]\n", r_i)
    print("A*[R] = H * G")
    print("  [I]:\n", (a @ r_i) % q)
    print("H * G:\n", (h_invertible_matrix @ g_primitive_matrix) % q)

    i_o = numpy.concatenate((numpy.identity(_m, dtype=int), numpy.zeros(shape=(w, _m), dtype=int)), axis=0)
    print("T':\n", i_o)
    t = numpy.concatenate((i_o, r_i), axis=1)
    print("T :\n", t)
    print("A@T:\n", (a @ t))
    return a, r_trapdoor


# Generating Shorter Bases for Hard Random Lattices
# Joel Alwen
# New York University
# Chris Peikert †
# Georgia Institute of Technology
# May 24, 2010

def hardTrapdoorGenerator(A1, n, m1, m2, l, q):
    m = m1 + m2
    G, R, U, P, C = frameworkTrapdoorGenerator(A1, n, m1, m2, l, q)
    # G, P = references.hsnf.Z_module.row_style_hermite_normal_form(A1)
    print("G")
    print(G)
    print("P")
    print(P)
    # U = numpy.array([[1, 0, 0],
    #                  [4, 2, -1],
    #                  [-3, -1, 1]])

    # G = numpy.array([[1, 0, 0],
    #                  [0, 1, 0],
    #                  [0, 0, 1]])
    #
    # P = numpy.array([[0, 1, 0],
    #                  [0, 1, 1],
    #                  [1, 0, 1]])

    # R = numpy.array([[0, 1, -1],
    #                  [0, -1, 0],
    #                  [1, 0, -1]])

    A2 = (-1 * numpy.array(A1)) @ (R + G)
    GR = G + R
    GRU = GR @ U
    GRU_RP = numpy.concatenate((GRU, (R @ P-C)), axis=1)
    print("GRU_RP")
    print(GRU_RP)
    UP = numpy.concatenate((U, P), axis=1)
    S = numpy.concatenate((GRU_RP, UP), axis=0)
    A = numpy.concatenate((A1, A2), axis=1)
    print("S")
    print(S)
    print("A2 % q")
    print(A2 % q)
    print("A2")
    print(A2)
    print("MERGE? <- ce ne trebuie")
    print(((A @ S) % q))

    print("TEST1: A1*(G + R)* U + A2 * U = 0")
    print((A1 @ (G + R)) @ U + A2 @ U)
    print("Test2: A1*(R*P - C) + A2 * P = 0")
    print(A1 @ (R @ P - C) + A2 @ P)
    print("A2")
    print(A2)
    print("P")
    print(P)
    print(len(R))
    print(len(R[0]))
    print(len(P))
    print(len(P[0]))
    # print((A1 @ (R @ P)) % q)

    print("Test3: A1 * (GP) = 0")
    return A, S


"""
[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  9  6  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  6  0  0  9]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  6  0  0  9]]
"""


def hermiteNormalFormAjtaiLattice(A1, m1, q):
    A1 = numpy.array(A1)
    Q = numpy.matrix([q]*len(A1)).T
    print("Q:", Q)
    A1 = numpy.concatenate((A1, Q), axis=1)
    solution = linsolve(Matrix(A1))
    substitution_list = []
    print("SYMPY SOL:", solution)
    for variable in solution.free_symbols:
        substitution_list.append((variable, randint(0, q//2)))
    print("SUBS LIST", substitution_list)
    print("SYMPY RESULT:", solution.subs(substitution_list))
    # TODO check this bug V list might be empty
    sol = [s % q for s in list(solution.subs(substitution_list))[0]]
    print(f"TRUE SOLUTION MOD {q} =", sol)
    H = numpy.zeros(shape=(m1, m1), dtype=int)
    H[0][0] = 1
    print(H)
    # q is prime => first h column is always q or 1 if there the column from A is 0 as h_i_i <= q
    if A1[:, 0].any():
        H[:, 0] *= q
    print(H)
    for j in range(1, m1):
        for i in range(1, j):
            for k in range(q // (j + 1), q + 1):
                H[j][j] = k

    return numpy.identity(n=m1, dtype=int) * q


def frameworkTrapdoorGenerator(A1, n, m1, m2, l, q):
    print(A1 @ numpy.zeros(shape=(m1, 1), dtype=int))
    H = hermiteNormalFormAjtaiLattice(A1, m1, q)
    # H = numpy.array([[11, 0, 0, 5, 4, 7],
    #                  [0, 11, 0, 5, 0, 9],
    #                  [0, 0, 11, 0, 0, 0],
    #                  [0, 0, 0, 6, 0, 2],
    #                  [0, 0, 0, 0, 7, 2],
    #                  [0, 0, 0, 0, 0, 10]])
    C = numpy.identity(n=m1, dtype=int)
    H_prim = H - C
    rng = numpy.random.default_rng()
    print(numpy.zeros(shape=(m1, l), dtype=int))
    if m2 - m1 * l != 0:
        G = numpy.zeros(shape=(m1, m2 - m1 * l), dtype=int)
    else:
        G = numpy.matrix()
    print("GGGGGG")
    print(G)
    for i in range(0, m1):
        g_i = numpy.zeros(shape=(m1, l), dtype=int)
        g_i[:, l - 1] = H_prim[:, i]
        for j in range(l - 2, 0, -1):
            g_i[:, j] = g_i[:, j + 1] // CONSTANT_R
        print("G_I")
        print(g_i)
        G = numpy.concatenate((g_i, G), axis=1)
    print("Final G")
    print(G)
    # P = numpy.identity(n=m1, dtype=int)
    # P = numpy.concatenate((P, numpy.zeros(shape=(m2-m1, m1), dtype=int)),axis=0)
    # print("P<- I_m1")
    # print(P)
    P = numpy.zeros(shape=(m2, m1), dtype=int)
    # P[l-1] = numpy.ones(shape=(1, m1), dtype=int)
    for _m1 in range(0, m1):
        print("_m1")
        print(_m1)
        P[((_m1+1)*l)-1][_m1] = 1
    P = numpy.flip(P, axis=1)
    print("FLIPPED P")
    print(P)
    print(G)
    print(P)
    print("G @ P:")
    print(G @ P)
    print("H'")
    print(H_prim)
    T = numpy.identity(n=l, dtype=int)
    triangle_index = numpy.triu_indices(l, 1)
    T[triangle_index] = -CONSTANT_R
    U = numpy.kron(numpy.eye(m1, dtype=int), T)
    zero1 = numpy.zeros(shape=(m1 * l, m2 - m1 * l), dtype=int)
    zero2 = numpy.zeros(shape=(m2 - m1 * l, m1 * l), dtype=int)
    miniI = numpy.identity(n=m2 - m1 * l, dtype=int)
    U = numpy.asarray(numpy.bmat([[U, zero1], [zero2, miniI]]))
    print("U:")
    print(U)
    print(len(U))
    print(len(U[0]))

    d = int(numpy.floor((1 + CONSTANT_SIGMA) * n * numpy.log10(q)))
    print(d)
    r_choices = numpy.array([0, 1, -1])
    R = numpy.random.choice(r_choices, size=(d, m2), p=[0.5, 0.25, 0.25], replace=True)
    R_zero = numpy.zeros(shape=(m1 - d, m2), dtype=int)
    R = numpy.concatenate((R, R_zero), axis=0)
    # print("D rows")
    # print(d)

    print(R)

    return G, R, U, P, C
    # for i in range(1, 100000):
    #     z = rng.integers(low=0, high=q - 1, size=(m1, 1), endpoint=True)
    #     if not ((A1 @ z) % q).any():
    #         print("FOUND!")
    #         print((A1 @ z) % q)
    #         print((A1 @ z))
    #         print("Z")
    #         print(z)
    #         print(A1)


# Based on
# Bonsai Trees, or How to Delegate a Lattice Basis
# David Cash?
# , Dennis Hofheinz??, Eike Kiltz? ? ?, and Chris Peikert†

def extendBasisRight(S_trapdoor, A_original_basis, _A_extension, prime_number):
    print("ENTERING EXTEND BASE")
    _A_extension = -_A_extension
    _A_extension %= prime_number
    print(_A_extension.T)
    W = []
    for column in _A_extension.T:
        print("Column,", column.T)
        linear_system = numpy.concatenate((A_original_basis, column.T), axis=1)
        solution = linsolve(Matrix(linear_system))
        solution_values = []
        for var_id in range(0, len(solution.free_symbols)):
            solution_values.append(("tau" + str(var_id), numpy.random.randint(0, prime_number + 1)))
        if len(W) == 0:
            W = numpy.matrix(list(list(solution.subs(solution_values))[0])).T
        else:
            W = numpy.concatenate((W, numpy.matrix(list(list(solution.subs(solution_values))[0])).T), axis=1)
        print("Column for W: \n", numpy.matrix(list(list(solution.subs(solution_values))[0])).T)
    print("W:\n", W)
    print("Checking A @ W:\n", A_original_basis @ W)
    print("Checking Ā:\n", _A_extension)
    SW = numpy.concatenate((S_trapdoor, W), axis=1)
    zeroMatrix = numpy.zeros(shape=(W.shape[1], W.shape[0]))
    identityMatrix = numpy.identity(n=W.shape[1], dtype=int)
    OI = numpy.concatenate((zeroMatrix, identityMatrix), axis=1)
    SW_OI = numpy.concatenate((SW, OI), axis=0)
    print("SW_OI\n", SW_OI)
    return SW_OI
