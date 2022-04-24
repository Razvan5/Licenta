import numpy as np
import constants.lattice_constants as lc
import lattice_utils.lattice_sampling as ls


def cut(lattice, tries, k):
    lc.PRIME = 11
    k = int(np.log2(lc.PRIME))
    x = ls.rejection_sampling(ls.discrete_gaussian_over_lattice, tries, lattice)
    print(x)
    return x[0]


def sample_preimage(lattice, trapdoor, uniform, s, sigma):
    pass


def approx_trap_gen():
    pass


def sigma_creator(R):
    superior_sigma = np.concatenate((R @ R.T, R.T), axis=1)
    inferior_sigma = np.concatenate((R, np.identity(n=R.shape[1], dtype=int)), axis=1)



# x = cut([[2, 0, 0, 0, 0],
#          [-1, 2, 0, 0, 0],
#          [0, -1, 2, 0, 0],
#          [0, 0, -1, 2, 1],
#          [0, 0, 0, -1, 1]], 100000, 1)
# print(x[0:5])
#
# print(np.exp((np.matrix([1, 3, 4]) @ np.matrix([[1, 5, 6], [7, 4, 1], [1, 0, 9]]) @ np.matrix([1, 3, 4]).T)))
#
# for i in range(100):
#     rng = np.random.binomial(10000000, 0.5)
#     print(rng)