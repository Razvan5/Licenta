# -------Custom Imports------- #
from constants import lattice_constants as lc
from lattice_utils.object_to_bits import unique_filename
from trapdoors import trapdoor as td
from lattice_utils import lattice_sampling as ls, policy as pl
from lattice_utils import object_to_bits as otb
from trapdoors import approximative_trapdoor as app_td
# -------Imports-------------- #
import os
import io
import sys
import time
import sympy
import secrets
import datetime
import numpy as np
import scipy as sy
from PIL import Image
from sympy import Matrix
from scipy import linalg
import matplotlib.pyplot as plt
from Cryptodome.Util.number import getPrime
from anytree import Node, AnyNode, RenderTree, AsciiStyle, LevelOrderGroupIter

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)


def kp_abe_setup(security_parameter, policy):
    lc.SECURITY_DIMENSION = getPrime(security_parameter)
    lc.PRIME = getPrime(security_parameter * 5)
    lc.ALL_ATTRIBUTES_LIMIT = policy.all_attribute_bound
    lc.KEY_ATTRIBUTES_LIMIT = policy.key_attribute_bound
    lc.M1 = int(np.floor((1 + lc.SIGMA) * lc.SECURITY_DIMENSION * np.log10(lc.PRIME)))
    lc.M2 = lc.M1 * lc.ALL_ATTRIBUTES_LIMIT + 1
    lc.LATTICE_DIMENSION = lc.M1 + lc.M2

    lattices_bases = []
    lattices_trapdoors = []
    print(f"n:{lc.SECURITY_DIMENSION} \nm1:{lc.M1} \nm2:{lc.M2} \nm:{lc.LATTICE_DIMENSION} \nq:{lc.PRIME} \n"
          f"attrs:{lc.ALL_ATTRIBUTES_LIMIT} \nkattrs:{lc.KEY_ATTRIBUTES_LIMIT}")

    for _ in range(1, lc.ALL_ATTRIBUTES_LIMIT+1):
        baseA, trapB = td.generateTrapdoor()
        lattices_bases.append(baseA)
        lattices_trapdoors.append(trapB)

    u = np.random.randint(low=0, high=lc.PRIME-1, size=lc.SECURITY_DIMENSION, dtype=int)
    print(f"random uniform u:\n{u}")
    A0 = np.random.randint(low=0, high=lc.PRIME-1, size=(lc.SECURITY_DIMENSION, lc.LATTICE_DIMENSION), dtype=int)
    print(f"random unform A0:\n{A0}")

    return (lattices_bases, A0, u), lattices_trapdoors


def kp_abe_extract(pub_key, ms_key, policy):
    L = policy.lsss_matrix
    print(f"L:\n{np.matrix(L)}")
    theta = policy.return_theta()
    attr_limit = policy.return_attribute_limit()
    Z = []
    for _ in range(0, theta-1):
        z = np.random.randint(low=0, high=lc.PRIME - 1, size=(lc.SECURITY_DIMENSION, lc.LATTICE_DIMENSION), dtype=int)
        Z.append(z)
    print(f"Z list: \n{Z}")
    lattice_bases, A0, u = pub_key
    diagonalAs = np.matrix(sy.linalg.block_diag(*lattice_bases))
    A0_wall = np.matrix(L[0][0] * A0)
    for i, l_row in enumerate(L):
        if i != 0:
            A0_wall = np.concatenate((A0_wall, l_row[0] * A0), axis=0)
    print(f"A0 Wall: \n {A0_wall}")

    Z_matrix = []
    for j in range(1, theta):
        Z_column = np.matrix(L[0][j] * Z[j-1])
        for i, l_row in enumerate(L):
            if i != 0:
                Z_column = np.concatenate((Z_column, l_row[1] * Z[0]), axis=0)
        print(f"Z_column {Z_column} {Z_column.shape}")
        if len(Z_matrix) == 0:
            Z_matrix = Z_column
        else:
            Z_matrix = np.concatenate((Z_column, Z_matrix), axis=1)
    M = np.concatenate((diagonalAs, A0_wall), axis=1)
    M = np.concatenate((M, Z_matrix), axis=1) % lc.PRIME
    diagonalBs = sy.linalg.block_diag(*ms_key)
    K = td.extendBasisRight(diagonalBs, diagonalAs, np.concatenate((A0_wall, Z_matrix), axis=1))
    print(f"M:{M.shape}\n{M}")
    print(f"K:{K.shape}\n{K}")
    print(f"M @ K:{M @ K}")
    print(f"Checking M @ K:{np.count_nonzero((M @ K) % lc.PRIME)}")
    return K


def kp_abe_encrypt(pub_key, attribs, msg):
    lattice_bases, A0, u = pub_key
    F = np.matrix(lattice_bases[0])
    for i in range(1, lc.ALL_ATTRIBUTES_LIMIT):
        F = np.concatenate((F, np.matrix(lattice_bases[i])), axis=1)
    F = np.concatenate((F, A0), axis=1)
    F = F % lc.PRIME

    print(f"F:{F.shape}\n{F}")
    s = np.random.randint(low=0, high=lc.PRIME-1, size=lc.SECURITY_DIMENSION, dtype=int)
    print(f"s uniform secret:\n{s}")
    v0 = np.random.randint(0, int(np.log10(lc.PRIME)))
    c0 = ((s.T @ u) + int(np.floor(lc.PRIME/2))*msg + v0) % lc.PRIME
    print(f"C0: {c0} (q={lc.PRIME})")
    # v1 = np.random.randint(low=0, high=1, size=F.shape[1], dtype=int)
    c1 = (s.T @ F) % lc.PRIME
    print(f"C1:{c1.shape}\n{c1}")
    return c0, c1


def kp_abe_decrypt(pub_key, key, ciphertext, attribs, policy):
    lattice_bases, A0, u = pub_key
    L = policy.lsss_matrix
    g = np.matrix(attribs)
    print(f"Checking g, {g @ L} ~ [d, 0, ... ,0]")
    d_value = (g @ L)[0, 0]
    K = key.copy()
    M_prim = []
    for i, attr in enumerate(attribs):
        if len(M_prim) == 0:
            M_prim = attr*lattice_bases[i]
        else:
            M_prim = np.concatenate(attr*lattice_bases[i])
    M_prim = np.concatenate((M_prim, d_value*A0), axis=1)
    print(f"M':{M_prim.shape}\n{M_prim}")
    M_second = []
    F_second = []
    for i, attr in enumerate(attribs):
        if attr != 0:
            if len(F_second) == 0:
                F_second = lattice_bases[i]
            else:
                F_second = np.concatenate((F_second, attr * lattice_bases[i]), axis=1)
        if attr != 0:
            if len(M_second) == 0:
                M_second = attr*lattice_bases[i]
            else:
                M_second = np.concatenate((M_second,attr*lattice_bases[i]), axis=1)
    M_second = np.concatenate((M_second, d_value*A0), axis=1)
    F_second = np.concatenate((F_second, A0), axis=1)
    print(f"M'':{M_second.shape}\n{M_second}")

    c0, c1 = ciphertext
    split_c1 = np.split(c1, lc.ALL_ATTRIBUTES_LIMIT+1, axis=1)
    print(split_c1)
    print(len(split_c1))
    split_k = np.split(K, lc.LATTICE_DIMENSION, axis=1)
    print("Split K:\n", split_k)
    print(len(split_k))
    c1_second = []
    K_second = []
    for i, attr in enumerate(attribs):
        if attr != 0:
            if len(K_second) == 0:
                K_second = split_k[i]
            else:
                K_second = np.concatenate((K_second, split_k[i]), axis=1)
            if len(c1_second) == 0:
                c1_second = split_c1[i]
            else:
                c1_second = np.concatenate((c1_second, split_c1[i]), axis=1)
    c1_second = np.concatenate((c1_second, split_c1[-1]), axis=1)
    K_second = np.concatenate((K_second, split_k[-1]), axis=1)
    print(f"C1'': {c1_second.shape},\n {c1_second}")
    print(f"K'': {K_second.shape},\n {K_second}")
    f_second, params, _ = Matrix(F_second).gauss_jordan_solve(Matrix(u), freevar=True)
    taus_zeroes = {tau: 0 for tau in params}
    f_second = f_second.xreplace(taus_zeroes)
    f_second = np.matrix(f_second)

    for i, line in enumerate(f_second):
        for j, element in enumerate(line):
            if type(element) is not int:
                element = element[0, 0]
                print(element)
                numerator, denominator = element.as_numer_denom()
                f_second[i, j] = (numerator*pow(denominator, -1, lc.PRIME))
    print(f"f'' {f_second.shape},\n{f_second}")
    print("Check F'' @ f'' :", F_second @ f_second % lc.PRIME, "==", u)
    v = c0 - f_second.T @ c1_second.T
    print(f"before modulo v : {v}")
    v = v % lc.PRIME
    print(f"{-np.floor(lc.PRIME/2)} <= v= {v} <= [{np.floor(lc.PRIME/2)}]")

    if abs(v) <= np.floor(lc.PRIME/4):
        return 0
    elif abs(v) >= np.ceil(lc.PRIME/4):
        return 1


if __name__ == '__main__':
    sys.stdout = open(unique_filename('console_output', '.txt'), 'w', encoding="utf-8")
    policy_tree = AnyNode(name="AND", tag=[1])
    or1 = AnyNode(name="OR", parent=policy_tree, tag=[])
    or2 = AnyNode(name="OR", parent=policy_tree, tag=[])
    a = AnyNode(name="A", parent=or1, tag=[])
    b = AnyNode(name="B", parent=or1, tag=[])
    c = AnyNode(name="C", parent=or2, tag=[])
    d = AnyNode(name="D", parent=or2, tag=[])
    # policy_tree = AnyNode(name="AND", tag=[1])
    # and1 = AnyNode(name="AND", parent=policy_tree, tag=[])
    # or1 = AnyNode(name="OR", parent=policy_tree, tag=[])
    # a = AnyNode(name="A", parent=and1, tag=[])
    # b = AnyNode(name="B", parent=and1, tag=[])
    # and2 = AnyNode(name="AND", parent=or1, tag=[])
    # d = AnyNode(name="D", parent=or1, tag=[])
    # f = AnyNode(name="F", parent=and2, tag=[])
    # g = AnyNode(name="G", parent=and2, tag=[])
    small_policy = pl.Policy(policy_tree, all_attribute_bound=4, key_attribute_bound=2)
    print(small_policy)
    start = time.time()
    public_key, master_key = kp_abe_setup(security_parameter=2, policy=small_policy)
    print("Setup Time:")
    print(time.time() - start)
    print(sys.getsizeof(public_key))
    print(sys.getsizeof(master_key))
    start2 = time.time()
    secret_key = kp_abe_extract(pub_key=public_key, ms_key=master_key, policy=small_policy)
    print("Extract Time:")
    print(time.time() - start2)
    print(sys.getsizeof(secret_key))
    cipher_attributes = [1, 1, 2, 0]
    percentage = 0
    tries = 100
    for i in range(0, tries):
        msg = np.random.randint(0, 2)
        print("Encrypt Time:")
        start3 = time.time()
        cipher_text = kp_abe_encrypt(pub_key=public_key, attribs=cipher_attributes, msg=msg)
        print(time.time() - start3)
        print(sys.getsizeof(cipher_text))

        start4 = time.time()
        decrypted_text = kp_abe_decrypt(
            pub_key=public_key,
            key=secret_key,
            ciphertext=cipher_text,
            attribs=cipher_attributes, policy=small_policy)
        if decrypted_text == msg:
            percentage += 1
        print("Decryption Time:")
        print(time.time() - start4)
        print(sys.getsizeof(decrypted_text))
        print(f"Decryption: {decrypted_text} == {msg} Message?")

    print("Percentage:", percentage/tries*100)
    bit_message = []
    print(otb.text_to_bits("Hi!"))
    for bit in otb.text_to_bits("Hi! My name is Razvan!"):
        msg = bit
        cipher_text = kp_abe_encrypt(pub_key=public_key, attribs=cipher_attributes, msg=msg)
        decrypted_text = kp_abe_decrypt(
            pub_key=public_key,
            key=secret_key,
            ciphertext=cipher_text,
            attribs=cipher_attributes, policy=small_policy)
        if decrypted_text is None:
            bit_message.append(0)
        else:
            bit_message.append(decrypted_text)
    print("Percentage:", percentage/tries*100)
    print(otb.text_from_bits(bit_message))
    sys.stdout.close()
