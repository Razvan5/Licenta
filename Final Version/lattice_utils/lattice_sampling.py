import numpy as np
from constants import lattice_constants as lc
from sympy import Matrix


# Cryptanalysis on ‘An efficient identity-based proxy
# signcryption using lattice’
# Zi-Yuan Liu, Yi-Fan Tseng, Raylin Tso*
# Department of Computer Science, National Chengchi University, Taipei 11605, Taiwan
# {zyliu, yftseng, raylin}@cs.nccu.edu.tw
# March 18, 2021
# https://eprint.iacr.org/2007/432.pdf
# Gaussian parameter s that was an exponential factor larger than the basis length kBk.


def gaussian_function(x, c, s):
    return np.exp((-np.pi * np.linalg.norm(x - c) ** 2) / s ** 2)


def discrete_gaussian_over_lattice(lattice, y, c=None, s=None):
    if s is None:
        s = np.linalg.norm(lattice)
    lattice_gaussian = 0
    c = np.zeros(shape=y.shape, dtype=int)
    for base_vector in lattice.T:
        lattice_gaussian += gaussian_function(base_vector, c, s)
    return gaussian_function(y, c, s) / lattice_gaussian

def discrete_gaussian_over_lattice_sigma_matrix(lattice, y, sigma_matrix, c=None, s=None):
    if s is None:
        s = np.linalg.norm(lattice)
    lattice_gaussian = 0
    c = np.zeros(shape=y.shape, dtype=int)
    for base_vector in lattice.T:
        lattice_gaussian += gaussian_function(base_vector, c, s)
    return gaussian_function(y, c, s) / lattice_gaussian


def rejection_sampling(pdf_function, maximum_val, lattice, return_immediately=True):
    vectors = [np.random.randint(0, lc.PRIME-1, size=np.matrix(lattice).shape[0]) for _ in range(0, maximum_val)]
    probabilities = np.random.random_sample(maximum_val+1)
    pdf_numbers = []
    pdf_probabilities = []
    for i, n in enumerate(vectors):
        print("-")
        if probabilities[i] < pdf_function(np.matrix(lattice), np.matrix(vectors[i])):
            if return_immediately:
                return n, probabilities[i], pdf_function(np.matrix(lattice), np.matrix(vectors[i]))
            pdf_numbers.append(n)
            pdf_probabilities.append(probabilities[i])
    return pdf_numbers


def find_shortest_vector(A1, A2, u):
    s2 = np.random.randint(low=0, high=lc.PRIME - 1, size=(1, lc.M2))
    s1, params, _ = Matrix(A1).gauss_jordan_solve(Matrix(u + A2 @ s2), freevar=True)
    taus_zeroes = {tau: 0 for tau in params}
    s1 = s1.xreplace(taus_zeroes)
    return np.concatenate(np.matrix(s1), s2) % lc.PRIME
