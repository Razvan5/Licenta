from sympy import *
from sympy.solvers.solveset import linsolve
import numpy
import time

h2, h1, h0 = symbols('h[2|2], h[1|2], h[0|2]')
print(linsolve(Matrix([[0, 0, 1, 11],
                       [1, 1, 0, 11],
                       [1, 1, 0, 11]]), [h2, h1, h0]))

h3, h2, h1, h0 = symbols('h[3|3], h[2|3], h[1|3], h[0|3]')
print(linsolve(Matrix([[1, 0, 1, 0, 11],
                       [1, 1, 0, 0, 11],
                       [1, 1, 0, 1, 11]]), [h3, h0, h1, h2]))

h4, h3, h2, h1, h0 = symbols('h[4|4], h[3|4], h[2|4], h[1|4], h[0|4]')
print(linsolve(Matrix([[0, 0, 1, 0, 1, 11],
                       [1, 1, 0, 0, 1, 11],
                       [1, 1, 0, 1, 1, 11]]), [h4, h0, h1, h2, h3]).subs([(h3, 8), (h0, -7)]))

x = linsolve(Matrix([[0, 0, 1, 0, 1, 11],
                       [1, 1, 0, 0, 1, 11],
                       [1, 1, 0, 1, 1, 11]]), [h4, h0, h1, h2, h3])

cond = And(h4 > h0, h4 > h1, h4 > h2, h4 > h3)
print(x.free_symbols)
print(x.free_symbols.pop())
fs = list(x.free_symbols)
print("__________________")
q = 3000
# [2900] elemente de ghicit corect + eleemntele sunt intre 0 si q-1 > 1 mil
# C 1million luat 2900
start = time.time_ns()
for i in range(0, q):
        numpy.array(list(x.subs([(fs[0], i), (fs[1], i)]))) % q
print("__________________")
print(time.time_ns() - start)
h5, h4, h3, h2, h1, h0 = symbols('h[5|5], h[4|5], h[3|5], h[2|5], h[1|5], h[0|5]')
print(linsolve(Matrix([[0, 0, 1, 0, 1, 0, 11],
                       [0, 1, 0, 0, 1, 1, 11],
                       [0, 1, 0, 1, 1, 1, 11]]), [h5, h0, h1, h2, h3, h4]).subs([(h5, 10), (h4, 2), (h3, 2)]))

