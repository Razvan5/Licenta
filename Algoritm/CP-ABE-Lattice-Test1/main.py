import math
import sys
from scipy.linalg import block_diag
from anytree import Node, AnyNode, RenderTree, AsciiStyle, LevelOrderGroupIter

# -custom imports-
from trapdoors.trapdoor import *
from trapdoors.sampling import *
from tests.random_prime import *
from custom_tree import *
import policy
numpy.set_printoptions(threshold=sys.maxsize)

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
    prime_modulus = generatePrimeNumber(4)
    print("PRIMEMODULUS:", prime_modulus)
    # m1
    global m1
    m1 = int(math.floor((1 + CONSTANT_SIGMA) * security_dimension * math.log10(prime_modulus)))
    # number of attributes
    global attribute_bound
    attribute_bound = int(math.ceil(math.log(prime_modulus, CONSTANT_R)))
    print("L, attribute bound", attribute_bound)
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
    print("n:", security_dimension)
    print("m:", lattice_dimension)
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
    L = _policy.lewko_waters_algorithm()
    Z = []
    Ai = _public_key[0]
    A0 = _public_key[1]
    u = _public_key[2]
    rng = numpy.random.default_rng()
    for i in range(0, _policy.return_theta()+1):
        zj = rng.integers(low=0, high=prime_modulus-1, size=(security_dimension, m1), endpoint=True)
        Z.append(zj)
    global attribute_bound
    print("Diagonal A's")
    diagonalAs = block_diag(*Ai)
    print(*Ai)
    print("LEN(AI)", len(Ai))
    print(diagonalAs)
    public_A0_wall = L[0][0]*A0
    # print("LEN(L):", len(L))
    for i in range(1, attribute_bound):
        public_A0_wall = numpy.concatenate((public_A0_wall, L[i][0]*A0), axis=0)
    print("A0:\n", A0.shape)
    print("A0 wall:\n", public_A0_wall)
    secret = []
    for i in range(0, attribute_bound):
        z_line = numpy.matrix(L[i][1]*Z[1])
        for j in range(2, _policy.return_theta()):
            print("L[i][j]*Z[j]")
            print(L[i][j]*Z[j])
            z_line = numpy.concatenate((z_line, L[i][j]*Z[j]), axis=1)
        if len(secret) == 0:
            secret = z_line
            print("LOLO:", secret)
        else:
            secret = numpy.concatenate((secret, z_line), axis=0)
    print("SECRET")
    print(secret)
    print("SHAPES:", diagonalAs.shape, public_A0_wall.shape, secret.shape)
    M = numpy.concatenate((diagonalAs, public_A0_wall), axis=1)
    M = numpy.concatenate((M, secret), axis=1)
    diagonalBs = block_diag(*_master_key)
    print("DIAG bi \n", diagonalBs)
    print("DIAGONAL MULTIPLIED TEST:", diagonalAs @ diagonalBs)
    print("M:\n", M)
    K = extendBasisRight(diagonalBs, diagonalAs, secret, prime_modulus)
    # print("M @ K", M @ K)
    return NotImplemented


def kp_abe_encrypt(_public_key, _attribute_list, _message_bit):
    # F is a encryption matrix
    Ai = _public_key[0]
    A0 = _public_key[1]
    u = _public_key[2]
    zeroMatrix = numpy.zeros(shape=A0.shape, dtype=int)
    if 0 in _attribute_list:
        F = Ai[0]
    else:
        F = zeroMatrix
    for index in range(1, len(Ai)):
        if index in _attribute_list:
            print(f"INDEX,{index}")
            F = numpy.concatenate((F, Ai[index]), axis=1)
        else:
            F = numpy.concatenate((F, zeroMatrix), axis=1)
    F = numpy.concatenate((F, A0), axis=1)
    F %= prime_modulus
    print("F\n", F)

    rng = numpy.random.default_rng()
    print("F, shape\n", F.shape)
    # s is a n vector uniformly random
    s = rng.integers(low=1, high=prime_modulus-1, size=(1, F.shape[0]), dtype=int)
    print(s)
    print(u)
    v0 = randint(1, 10000000)
    c0 = (numpy.squeeze(s @ u) +  + int(numpy.floor(prime_modulus/2))*_message_bit) % prime_modulus
    print("CO", c0)
    # TODO add distribution
    v1 = randint(1, 10000000)
    c1 = (s @ F + v1) % prime_modulus
    print("C1", c1)
    Ctx = (c0, c1)
    return Ctx


def kp_abe_decrypt(_public_key, _key, _ciphertext_bit):
    return NotImplemented


if __name__ == '__main__':
    policy_tree1 = AnyNode(name="AND", tag=[1])
    e = AnyNode(name="E", parent=policy_tree1, tag=[])
    or1 = AnyNode(name="OR", parent=policy_tree1, tag=[])
    or2 = AnyNode(name="OR", parent=or1, tag=[])
    and1 = AnyNode(name="AND", parent=or1, tag=[])
    or3 = AnyNode(name="OR", parent=and1, tag=[])
    a1 = AnyNode(name="A", parent=or3, tag=[])
    b1 = AnyNode(name="B", parent=or3, tag=[])
    or4 = AnyNode(name="OR", parent=and1, tag=[])
    c1 = AnyNode(name="C", parent=or4, tag=[])
    d2 = AnyNode(name="D", parent=or4, tag=[])
    and2 = AnyNode(name="AND", parent=or2, tag=[])
    a = AnyNode(name="A", parent=and2, tag=[])
    b = AnyNode(name="B", parent=and2, tag=[])
    and3 = AnyNode(name="AND", parent=or2, tag=[])
    c = AnyNode(name="C", parent=and3, tag=[])
    d = AnyNode(name="D", parent=and3, tag=[])

    p = policy.Policy(policy_tree1, 0)
    p.lewko_waters_algorithm()

    x, y = kp_abe_setup(2, 2)
    print("TESTING ALL BASES:\n", x[0][0] @ y[0])

    # TODO extendRight in that cursed book
    print(kp_abe_extract(x, y, p))

    # TODO 0, 1, 2 sunt [E, A, B], unul din seturile care dau corect pt policy-ul de mai sus
    crypto_text = kp_abe_encrypt(_public_key=x, _attribute_list=[0, 1, 2], _message_bit=1)



    # policy_tree = AnyNode(name="AND", tag=[1])
    # a5 = AnyNode(name="A5", parent=policy_tree, tag=[])
    # or1 = AnyNode(name="OR", parent=policy_tree, tag=[])
    # and1 = AnyNode(name="AND", parent=or1, tag=[])
    # a1 = AnyNode(name="A1", parent=and1, tag=[])
    # a2 = AnyNode(name="A2", parent=and1, tag=[])
    # and2 = AnyNode(name="AND", parent=or1, tag=[])
    # a3 = AnyNode(name="A3", parent=and2, tag=[])
    # a4 = AnyNode(name="A4", parent=and2, tag=[])

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
