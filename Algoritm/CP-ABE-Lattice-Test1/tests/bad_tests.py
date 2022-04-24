from sympy import *
from sympy.solvers.solveset import linsolve
import numpy
import time

import prints_tests
# h2, h1, h0 = symbols('h[2|2], h[1|2], h[0|2]')
# print(linsolve(Matrix([[0, 0, 1, 11],
#                        [1, 1, 0, 11],
#                        [1, 1, 0, 11]]), [h2, h1, h0]))
#
# h3, h2, h1, h0 = symbols('h[3|3], h[2|3], h[1|3], h[0|3]')
# print(linsolve(Matrix([[1, 0, 1, 0, 11],
#                        [1, 1, 0, 0, 11],
#                        [1, 1, 0, 1, 11]]), [h3, h0, h1, h2]))
#
# h4, h3, h2, h1, h0 = symbols('h[4|4], h[3|4], h[2|4], h[1|4], h[0|4]')
# print(linsolve(Matrix([[0, 0, 1, 0, 1, 11],
#                        [1, 1, 0, 0, 1, 11],
#                        [1, 1, 0, 1, 1, 11]]), [h4, h0, h1, h2, h3]).subs([(h3, 8), (h0, -7)]))
#
# x = linsolve(Matrix([[0, 0, 1, 0, 1, 11],
#                        [1, 1, 0, 0, 1, 11],
#                        [1, 1, 0, 1, 1, 11]]), [h4, h0, h1, h2, h3])
#
# cond = And(h4 > h0, h4 > h1, h4 > h2, h4 > h3)
# print(x.free_symbols)
# print(x.free_symbols.pop())
# fs = list(x.free_symbols)
# print("__________________")
# q = 3000
# # [2900] elemente de ghicit corect + eleemntele sunt intre 0 si q-1 > 1 mil
# # C 1million luat 2900
# start = time.time_ns()
# for i in range(0, q):
#         numpy.array(list(x.subs([(fs[0], i), (fs[1], i)]))) % q
# print("__________________")
# print(time.time_ns() - start)
# h5, h4, h3, h2, h1, h0 = symbols('h[5|5], h[4|5], h[3|5], h[2|5], h[1|5], h[0|5]')
# print(linsolve(Matrix([[0, 0, 1, 0, 1, 0, 11],
#                        [0, 1, 0, 0, 1, 1, 11],
#                        [0, 1, 0, 1, 1, 1, 11]]), [h5, h0, h1, h2, h3, h4]).subs([(h5, 10), (h4, 2), (h3, 2)]))

from random import randint


def getHermiteColumn(linear_system, prime_number, column_index, hermite_matrix_size):
    solution = linsolve(Matrix(linear_system))
    possible_column = numpy.array([])
    last_norm = 999999999
    if len(list(solution)) == 0:
        e = [0] * hermite_matrix_size
        e[-column_index - 1] = prime_number
        return (numpy.matrix(e)).T

    has_finite_solution = True
    for alfa_object in list(list(solution))[0]:
        if not (str(alfa_object)).isnumeric():
            has_finite_solution = False
            break

    if has_finite_solution:
        numbered_solution = list(list(solution)[0])
        numbered_solution = [n_s % prime_number for n_s in numbered_solution]
        max_value = max(numbered_solution)
        if numbered_solution.index(max_value) != len(numbered_solution) - 1:
            e = [0] * hermite_matrix_size
            e[-column_index - 1] = prime_number
            return (numpy.matrix(e)).T
    elif has_finite_solution:
        numbered_solution = list(list(solution)[0])
        numbered_solution = [n_s % prime_number for n_s in numbered_solution]
        max_value = max(numbered_solution)
        if numbered_solution.index(max_value) == len(numbered_solution) - 1:
            return (numpy.array(numbered_solution)).T

    print("Matrix Recv:\n", linear_system)
    solution_free_var_len = len(solution.free_symbols)
    print("Alpha Solution", solution)
    for _ in range(0, 50):
        solution_values = []
        for var_id in range(0, solution_free_var_len):
            solution_values.append(("tau" + str(var_id), randint(0, prime_number + 1)))
        numbered_solution = list(list(solution.subs(solution_values))[0])
        numbered_solution = [n_s % prime_number for n_s in numbered_solution]
        print("Possible Solution", numbered_solution)
        max_value = max(numbered_solution)
        check_for_fractions = False
        for numeric in numbered_solution:
            if not str(numeric).isnumeric():
                check_for_fractions = True
                break
        if numbered_solution.index(max_value) == len(numbered_solution) - 1 and check_for_fractions is False:
            current_norm = numpy.linalg.norm(numpy.array(numbered_solution).astype(numpy.int64))
            print("Current Norm:", current_norm)
            if current_norm < last_norm:
                last_norm = current_norm
                print("100% Solution:", numbered_solution)
                numbered_solution += [0] * (hermite_matrix_size - len(numbered_solution))
                possible_column = numpy.array(numbered_solution)

    # m = len(solution.free_symbols)
    # print(f"SOLUTION:{solution}")
    # print(f"SOLUTION2:{len(list(solution))}")
    # print(f"SOLUTION3:{len(list(solution)) == 0}")
    # if len(list(solution)) == 0:
    #     e = [0] * _m
    #     e[-_i-1] = 11
    #     return (numpy.matrix(e)).T
    # print("GET COLUMN ", _i, " FOR ", solution)
    # print(linear_system)
    # for i in range(1, 30):
    #     s = []
    #     for j in range(0, m):
    #         s.append(("tau" + str(j), randint(0, _q + 1)))
    #     s_list = list(solution.subs(s))[0]
    #     s_list = [number % _q for number in s_list]
    #     print(f'_m:{_m} len(list){len(s_list)} = {_m - len(s_list)} -> '
    #           f'{[0] * (_m - len(s_list))}')
    #     s_list += [0] * (_m - len(s_list))
    #     print("List" + str(i))
    #     print(s_list)
    #     is_correct = True
    #     for k in range(0, len(s_list) - _i - 1):
    #         if s_list[-_i-1] <= s_list[k]:
    #             print(f"LOL: {s_list}, s_list({len(s_list)-_i-1}){s_list[-_i-1]} < s_list({k}){s_list[k]}")
    #             is_correct = False
    #             break
    #     if is_correct:
    #         print("IT IS CORRECT!")
    #         current_norm = numpy.linalg.norm(numpy.array(s_list).astype(numpy.int64))
    #         print("Vector:", numpy.array(s_list).astype(numpy.int64))
    #         print("Norm:", current_norm)
    #         if last_norm > current_norm:
    #             possible_column = numpy.array(s_list).astype(numpy.int64)
    #             last_norm = current_norm
    #
    # print("POS_COL->", numpy.matrix(possible_column).T)
    if possible_column.any():
        return numpy.matrix(possible_column).T
    else:
        e = [0] * hermite_matrix_size
        e[-column_index - 1] = prime_number
        return (numpy.matrix(e)).T


def hermiteForm(uniform_matrix, prime_number):
    changed_matrix = uniform_matrix.copy()
    n = uniform_matrix.shape[0]
    m = uniform_matrix.shape[1]
    Q_column = numpy.matrix([prime_number] * n).T
    uniform_matrix = numpy.concatenate((changed_matrix, Q_column), axis=1)
    H = getHermiteColumn(uniform_matrix, prime_number, 0, m)
    for i in range(1, m):
        changed_matrix = numpy.delete(changed_matrix, -1, 1)
        uniform_matrix = numpy.concatenate((changed_matrix, Q_column), axis=1)
        column = getHermiteColumn(uniform_matrix, prime_number, i, m)
        H = numpy.concatenate((column, H), axis=1)
    return H


if __name__ == '__main__':
    # solution = linsolve(Matrix([[0, 0, 1, 0, 1, q],
    #                             [1, 1, 0, 0, 1, q],
    #                             [1, 1, 0, 1, 1, q]]))
    # print(solution)
    #
    # for i in range(1, 10):
    #     s = []
    #     for j, tau in enumerate(solution.free_symbols):
    #         # t = symbols("tau"+str(j), positive=True, rational=False, natural=True)
    #         # print(t.is_Number)
    #         s.append(("tau"+str(j), randint(0, q+1)))
    #     print(solution.subs(s))
    # print("___________________________")
    # q = 11
    # a1, a2, a3 = symbols("a0, a1, a2")
    # solution = linsolve(Matrix([[0, 0, 1, 0, 1, 0, q],
    #                             [0, 1, 0, 0, 1, 1, q],
    #                             [0, 1, 0, 1, 1, 1, q]]))
    # print(solution)
    #
    # for i in range(1, 30):
    #     s = []
    #     for j, tau in enumerate(solution.free_symbols):
    #         t = symbols("tau"+str(j), positive=True, rational=False, natural=True)
    #         # print(t.is_Number)
    #         s.append(("tau"+str(j), randint(0, q+1)))
    #     print("List"+str(i))
    #     s_list = list(solution.subs(s))[0]
    #     is_correct = True
    #     for k in range(1, len(s_list)):
    #         if s_list[0] % q <= s_list[k] % q:
    #             is_correct = False
    #     if is_correct:
    #         for k in range(1, len(s_list)):
    #             print(s_list[k] % q)
    #         print(s_list[0])
    #         print("NORMA:", numpy.linalg.norm(numpy.array(s_list).astype(numpy.int64)))
    #         print()
    q = 11
    A = numpy.matrix(
        [[0, 1, 0, 1, 0, 0],
         [1, 0, 0, 1, 1, 0],
         [1, 0, 1, 1, 1, 0]])
    rng = numpy.random.default_rng()
    A = rng.integers(low=0, high=1, size=(3, 9), endpoint=True)
    # print(A)
    # L = [1, 2, 3, 4, 5]
    # print(max(L))
    # x = max(L)
    # print(L.index(x))
    H = hermiteForm(A, q)
    print("A\n", A)
    print("H\n", H)
    print("A*H:\n", A @ H)
    print((A @ H) % q)
    # print(getHermiteColumn(A, 11, 3))
