def matrix_print(matrix):
    print("+" + "-" * 2 * len(matrix[0]) + "+")
    for line in matrix:
        print("|", end="")
        for element in line:
            print(element, end=" ")
        print("|")
    print("+" + "-" * 2 * len(matrix[0]) + "+")


def full_print(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)
