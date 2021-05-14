import references.hsnf.Z_module
import trapdoor
import random_prime
from lattice_constants import *
from references.hsnf.Z_module import ZmoduleHomomorphism
import math
import numpy
from prints_tests import matrix_print

# z = references.hsnf.Z_module.row_style_hermite_normal_form(M=[[1, 0, 1],
#                                                               [1, 1, 3],
#                                                               [4, 1, 8]
#                                                               ])
# print(z[0])
# print(z[1])
# 248 -> [[1, 0, 1],
#         [20, 1, 3],
#         [4, 11, 65]]
# x = ZmoduleHomomorphism(A=[[1, 0, 1],
#                            [20, 1, 3],
#                            [4, 11, 65]], basis_from=(10, ), basis_to=(10, ))
#
# print(x.hermite_normal_form()[0])
# print(x.hermite_normal_form()[1])

# trapdoor.hardTrapdoorGenerator([[1, 0, 1], [1, 1, 3], [4, 1, 8]], 2, 2, 3, 11)

# a1 = references.hsnf.Z_module.row_style_hermite_normal_form(
#     [[1, 0, 1, 1],
#      [1, 1, 3, 3],
#      [4, 1, 8, 5]
#      ])
# a2 = references.hsnf.Z_module.row_style_hermite_normal_form(
#     [[1, 0, 1, 1],
#      [1, 1, 3, 3],
#      [4, 1, 8, 5],
#      [0, 0, 0, 0]
#      ])
#
# print(a1[0])
# print(a1[1])
# print(a2[0])
# print(a2[1])

n = 3
q = 7919
m1 = math.floor((1 + CONSTANT_SIGMA) * n * math.log10(q))
l = math.ceil(math.log(q, CONSTANT_R))
m2 = m1 * l + 1
print(
    "n" + " " * (len(str(q))) + "q" + " " * (len(str(l))) + "l" + " " * (len(str(m1))) + "m1" +
    " " * (len(str(m2))) + "m2" + " " * (len(str(m1 + m2))) + "m")
print(n, q, l, " " + str(m1), " " + str(m2), m1 + m2)

rng = numpy.random.default_rng()
A1 = rng.integers(low=0, high=1, size=(n, m1), endpoint=True)
# A1 = numpy.array([[0, 1, 0, 1, 0, 0],
#                   [1, 0, 0, 1, 1, 0],
#                   [1, 0, 1, 1, 1, 0]])
print("A1")
matrix_print(A1)
# trapdoor.hermiteNormalFormAjtaiLattice(A1, m1, q)
# G, R, U, P, C = trapdoor.frameworkTrapdoorGenerator(A1, n, m1, m2, l, q)
# A, S = trapdoor.hardTrapdoorGenerator(A1, n, m1, m2, l, q)
theta = 100
x = "{0:0" + str(theta) + "b}"
print(x)
print(numpy.array(list(x.format(100)), dtype=int))

from sympy import *
from sympy.solvers.solveset import linsolve

text = ""
for i in range(1, 6):
    text += f"h{str(i)}, "

print(text)
symbols_x = symbols(text)
# print(linsolve(Matrix(rng.integers(low=0, high=1, size=(n, m1), endpoint=True))))
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
solution = linsolve(Matrix([[1, 1, 1, 11], [1, 0, 1, 11], [1, 1, 1, 11]]), [x, y, z])
print(solution)

cond = And(0 < x, 0 < z, 0 < y, y < x, z < x, x <= 11)
print((solution.subs(z, 1)))

h2, h1, h0 = symbols('h[2|2], h[1|2], h[0|2]')
print(linsolve(Matrix([[1, 0, 1, 11],
                       [1, 1, 0, 11],
                       [1, 1, 0, 11]]), [h2, h1, h0]))

h3, h2, h1, h0 = symbols('h[3|3], h[2|3], h[1|3], h[0|3]')
print(linsolve(Matrix([[1, 0, 1, 0, 11],
                       [1, 1, 0, 0, 11],
                       [1, 1, 0, 1, 11]]), [h3, h0, h1, h2]))

h3, h2, h1, h0 = symbols('h[3|3], h[2|3], h[1|3], h[0|3]')
print(linsolve(Matrix([[1, 0, 1, 0, 11],
                       [1, 1, 0, 0, 11],
                       [1, 1, 0, 1, 11]]), [h3, h0, h1, h2]).subs(h3, 5))


h4, h3, h2, h1, h0 = symbols('h[4|4], h[3|4], h[2|4], h[1|4], h[0|4]')
print(linsolve(Matrix([[1, 0, 1, 0, 11],
                       [1, 1, 0, 0, 11],
                       [1, 1, 0, 1, 11]]), [h4, h0, h1, h2, h3]).subs([(h3, 6), (h1, 2)]))

