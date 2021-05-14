import numpy
from lattice_constants import CONSTANT_R


class Policy:
    def __init__(self, attribute_list_tuple, attribute_bound):
        self.attribute_list = attribute_list_tuple
        self.attribute_bound = attribute_bound

    def __repr__(self):
        return f"Policy<{len(self.attribute_list)}:{self.attribute_list}>"

    def __str__(self):
        str_object = ""
        str_arrows = ""
        str_binary = ""
        for attr in self.attribute_list:
            s = str(attr[0])
            str_object = str_object + s + " "
            str_arrows = str_arrows + (len(s)-len(str(attr[1])))*" " + "V "
            str_binary = str_binary + (len(s)-len(str(attr[1])))*" " + str(attr[1]) + " "
        str_object += "\n"
        str_arrows += "\n"
        return "Policy:\n" + str_object + str_arrows + str_binary

    def return_theta(self):
        return len(str(bin(self.attribute_bound)))

    def policy_to_linear_span_matrix(self):
        # span_program_matrix_L in Z^(l×(1+θ))
        theta = len(str(bin(self.attribute_bound)))
        span_program_matrix_L = numpy.zeros(shape=(self.attribute_bound, theta), dtype=int)
        e_1 = numpy.zeros(shape=(1, theta), dtype=int)
        e_1[0] = 1

        # TODO binary string care are mereu 0 ca prim element <-test if it holds! <- doesn't remove it (Matrice M va avea )
        temp = "{0:0" + str(theta) + "b}"
        # print(temp)
        for index, (attribute, exists) in enumerate(self.attribute_list):
            sign = -1
            incrementer = 1
            if exists:
                bin_attr = numpy.array(list(temp.format(attribute)), dtype=int)
                e_1 = numpy.add(e_1,  incrementer*bin_attr)
                incrementer += 1
                span_program_matrix_L[attribute] = bin_attr
        span_program_matrix_L[0] = e_1
        return span_program_matrix_L




