import numpy
import sympy
import math
import random
from matplotlib import pyplot as mp
# -custom imports-
from trapdoor import *
from sampling import *
from random_prime import *
from prints_tests import *

SMALL_NORM = 100


def kp_abe_setup(_lambda, _l):
    """

    :param _lambda: -> security parameter
    :param _l: -> attribute bound
    :return: -> public and master key
    """
    _lambda = int("1" * _lambda, 2)
    # n
    security_dimension = generatePrimeNumber(_lambda)
    # q <- the bigger the modulus the weaker the assumption of LWE
    prime_modulus = generatePrimeNumber(4)
    # m
    lattice_dimension = 2 * security_dimension * int(math.log2(prime_modulus)) + random.randint(1, 10)
    # TrapGen(1^Λ), for i ∈ [l] -> A_i ∈ (Z_q)^(n×m) cu B_i m-vector ⊆ (Λ_q)^⊥(A_i)
    lattice_bases_a = []
    vector_set_b = []
    # print("n:", security_dimension)
    # print("m:", lattice_dimension)
    for i in range(1, _l + 1):
        a, b = trapdoorGenerator(security_dimension, lattice_dimension, prime_modulus)
        lattice_bases_a.append(a)
        vector_set_b.append(b)
        # lattice_bases_a[i], vector_set_b[i] = trap_gen(_lambda, security_dimension, lattice_dimension)
    # A0 ∈ (Z_q)^(n×m)
    random_base_a0 = [random.randint(1, prime_modulus)
                      for _ in range(1, int(security_dimension + 1))
                      for _ in range(1, int(lattice_dimension + 1))]
    uniform_random_vector = [random.randint(1, prime_modulus) in range(1, security_dimension + 1)]

    public_key = (lattice_bases_a, random_base_a0, uniform_random_vector)
    master_key = vector_set_b

    return public_key, master_key


def kp_abe_extract(_public_key, _master_key, _policy):
    pass


def kp_abe_encrypt(_public_key, _attribute_list, _message):
    pass


def kp_abe_decrypt(_public_key, _key, _ciphertext):
    pass


if __name__ == '__main__':
    # x, y = kp_abe_setup(2, 3)
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
