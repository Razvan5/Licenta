import numpy
import sympy
import math
import random
from matplotlib import pyplot as mp
from scipy.linalg import block_diag

# -custom imports-
from trapdoor import *
from sampling import *
from random_prime import *
from prints_tests import *
import policy

attribute_bound = None
security_dimension = None
m1 = None
m2 = None
lattice_dimension = None
prime_modulus = None


def kp_abe_setup(_lambda, _l):
    """

    :param _lambda: -> security parameter
    :param _l: -> attribute bound
    :return: -> public and master key
    """
    _lambda = int("1" * _lambda, 2)
    # n
    global security_dimension
    security_dimension = generatePrimeNumber(_lambda)
    # q <- the bigger the modulus the weaker the assumption of LWE
    global prime_modulus
    prime_modulus = generatePrimeNumber(9)
    print("PRIMEMODULUS:", prime_modulus)
    # m1
    global m1
    m1 = int(math.floor((1 + CONSTANT_SIGMA) * security_dimension * math.log10(prime_modulus)))
    # number of attributes
    global attribute_bound
    attribute_bound = int(math.ceil(math.log(prime_modulus, CONSTANT_R)))
    # m2
    global m2
    m2 = m1 * attribute_bound + 1
    # m
    global lattice_dimension
    lattice_dimension = m1 + m2
    # lattice_dimension = 2 * security_dimension * int(math.log2(prime_modulus))
    # TrapGen(1^Λ), for i ∈ [l] -> A_i ∈ (Z_q)^(n×m) cu B_i m-vector ⊆ (Λ_q)^⊥(A_i)
    lattice_bases_a = []
    vector_set_b = []
    # print("n:", security_dimension)
    # print("m:", lattice_dimension)
    rng = numpy.random.default_rng()
    for i in range(1, attribute_bound + 1):
        A1 = rng.integers(low=0, high=1, size=(security_dimension, m1), endpoint=True)
        a, b = hardTrapdoorGenerator(A1, security_dimension, m1, m2, attribute_bound, prime_modulus)
        lattice_bases_a.append(a)
        vector_set_b.append(b)
        # lattice_bases_a[i], vector_set_b[i] = trap_gen(_lambda, security_dimension, lattice_dimension)
    # A0 ∈ (Z_q)^(n×m)
    rng = numpy.random.default_rng()
    random_base_a0 = rng.integers(low=0, high=prime_modulus-1, size=(security_dimension, lattice_dimension), dtype=int)
    uniform_random_vector = rng.integers(low=0, high=prime_modulus-1, size=(security_dimension, 1), dtype=int)
    print("UNIFORM")
    print(uniform_random_vector)
    public_key = (lattice_bases_a, random_base_a0, uniform_random_vector)
    master_key = vector_set_b

    return public_key, master_key


def kp_abe_extract(_public_key, _master_key, _policy):
    L = _policy.policy_to_linear_span_matrix()
    Z = []
    rng = numpy.random.default_rng()
    for i in range(0, _policy.return_theta()):
        zj = rng.integers(low=0, high=prime_modulus-1, size=(security_dimension, m1), endpoint=True)
        Z.append(zj)
    global attribute_bound
    print("Diagonal A's")
    diagonalAs = block_diag(*_public_key[0])
    print(diagonalAs)


def kp_abe_encrypt(_public_key, _attribute_list, _message_bit):
    pass


def kp_abe_decrypt(_public_key, _key, _ciphertext_bit):
    pass


if __name__ == '__main__':
    p = policy.Policy([(1, 1), (6, 1), (3, 1)], 9)
    print(p.policy_to_linear_span_matrix())
    print(p)
    x, y = kp_abe_setup(2, 3)
    print(x[0][0] @ y[0])
    print(attribute_bound)

    print(kp_abe_extract(x, y, p))
    # p = 5
    # x = numpy.array([[2, 1],
    #                  [1, 2]])
    # y = x.__invert__()
    # z = (x @ y)
    # print(z)
    # print(z % 5)

    # g = [1, 2, 4, 8, 16]
    # i = numpy.identity(5, dtype=int)
    # G = numpy.kron(i, g)
    # matrix_print(G)

    # rng = numpy.random.default_rng()
    # zero = numpy.zeros(shape=(3, 1), dtype=int)
    # q = 11
    # for i in range(100):
    #     m = rng.integers(low=0, high=10, size=(4, 3), endpoint=True)
    #     print("M:")
    #     matrix_print(m)
    #     for j in range(100):
    #         v = rng.integers(low=2, high=10, size=(1, 4), endpoint=True)
    #         print("v:")
    #         matrix_print(v)
    #         print("m @ v:")
    #         mv = (v @ m) % q
    #         if not mv.any():
    #             print("FOUND!\n")
    #             print("~~~~~~~~~")
    #             matrix_print(v)
    #             matrix_print(m)
    #             print("@@@@@@@@")
    #             matrix_print(mv)
    #             print("~~~~~~~~~")
    #
    #             break

    # matrix with multiples of q
    # q = 11
    #
    # rng = numpy.random.default_rng()
    # A = rng.integers(low=1, high=10, size=(4, 3), endpoint=True)
    #
    # for i in range(10):
    #     fake_zero_array = rng.integers(low=1, high=q-1, size=(1, 3), endpoint=True)
    #     matrix_print(fake_zero_array * q)
    #
    #     x = fake_zero_array @ numpy.linalg.inv(A)
    #     matrix_print(A)
    #     print("X:")
    #     matrix_print(x)
    #     matrix_print(x % q)
    #     print("ANSWER")
    #     matrix_print((x @ A))
    #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
