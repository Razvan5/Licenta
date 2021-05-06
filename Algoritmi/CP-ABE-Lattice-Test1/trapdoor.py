import numpy
#
import lattice_constants
from sampling import randomVectorFromMatrix

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
    w = numpy.random.randint(2, m-1)
    print("w:", w)
    _m = m - w
    print("_")
    print("m:", _m)
    if _a_random_matrix is None:
        _a_random_matrix = numpy.random.randint(low=0, high=q-1, size=(n, _m))
        print("_A:\n", _a_random_matrix)
    if h_invertible_matrix is None:
        # trisant has to be something different
        h_invertible_matrix = numpy.identity(n, dtype=int)
        h_invertible_matrix = numpy.random.randint(low=0, high=q-1, size=(n, n))
        # needs invertible check! (if det != 0)
        print("INVERTIBLE MATRIX:\n", h_invertible_matrix)

    rng = numpy.random.default_rng()
    r_trapdoor = rng.integers(low=0, high=q-1, size=(_m, w), endpoint=True)
    print("TRAPDOOR:\n", r_trapdoor)
    g_primitive_matrix = rng.integers(low=1, high=q-1, size=(n, w))
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
