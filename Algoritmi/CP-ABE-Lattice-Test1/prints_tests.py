def matrix_print(matrix):
    print("+"+"-"*2*len(matrix[0])+"+")
    for line in matrix:
        print("|",end="")
        for element in line:
            print(element, end=" ")
        print("|")
    print("+"+"-"*2*len(matrix[0])+"+")
