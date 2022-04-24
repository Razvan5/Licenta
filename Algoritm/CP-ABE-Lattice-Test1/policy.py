import numpy
from anytree import Node, AnyNode, RenderTree, AsciiStyle, LevelOrderGroupIter


class Policy:
    def __init__(self, attribute_tree, attribute_bound):
        self.attribute_tree = attribute_tree
        self.attribute_bound = attribute_bound
        self.theta = -1

    def __repr__(self):
        return self.attribute_tree.__repr__

    def __str__(self):
        return self.attribute_tree.__str__

    def return_theta(self):
        return self.theta

    def lewko_waters_algorithm(self):
        for pre, fill, node in RenderTree(self.attribute_tree):
            print("%s%s %s" % (pre, node.name, node.tag))
        policy_tree_levels = [[node for node in children] for children in LevelOrderGroupIter(self.attribute_tree)]
        c = 1
        print(policy_tree_levels)
        for level in policy_tree_levels:
            # print(level)
            for node in level:
                # print(node)
                # print(c)
                if node.parent is not None and node.children is not None and node.name == "OR":
                    node.children[0].tag = node.tag.copy()
                    node.children[1].tag = node.tag.copy()
                elif node.children is not None and node.name == "AND":
                    while len(node.tag) != c:
                        node.tag.append(0)
                    if not node.is_leaf:
                        assert node.children is not None
                        node.children[1].tag = node.tag.copy()
                        node.children[1].tag.append(1)
                        node.children[0].tag = ([0] * c)
                        node.children[0].tag.append(-1)
                        c += 1

        lsss_matrix = []
        for pre, _, node in RenderTree(self.attribute_tree):
            if node.name != "AND" and node.name != "OR":
                node.tag += [0]*(c - len(node.tag))
                lsss_matrix.append(node.tag)
            print("%s%s %s" % (pre, node.name, node.tag))
        print(f"LSSS MATRIX:")
        for v in lsss_matrix:
            print(v)
        self.theta = len(lsss_matrix[0])
        return lsss_matrix


# class Policy:
#     def __init__(self, attribute_list_tuple, attribute_bound):
#         self.attribute_list = attribute_list_tuple
#         self.attribute_bound = attribute_bound
#
#     def __repr__(self):
#         return f"Policy<{len(self.attribute_list)}:{self.attribute_list}>"
#
#     def __str__(self):
#         str_object = ""
#         str_arrows = ""
#         str_binary = ""
#         for attr in self.attribute_list:
#             s = str(attr[0])
#             str_object = str_object + s + " "
#             str_arrows = str_arrows + (len(s)-len(str(attr[1])))*" " + "V "
#             str_binary = str_binary + (len(s)-len(str(attr[1])))*" " + str(attr[1]) + " "
#         str_object += "\n"
#         str_arrows += "\n"
#         return "Policy:\n" + str_object + str_arrows + str_binary
#
#     def return_theta(self):
#         return len(str(bin(self.attribute_bound)))
#
#     def policy_to_linear_span_matrix(self):
#         # span_program_matrix_L in Z^(l×(1+θ))
#         theta = len(str(bin(self.attribute_bound)))
#         span_program_matrix_L = numpy.zeros(shape=(self.attribute_bound, theta), dtype=int)
#         e_1 = numpy.zeros(shape=(1, theta), dtype=int)
#         e_1[0] = 1
#
#         # TODO binary string care are mereu 0 ca prim element <-test if it holds! <- doesn't remove it (Matrice M va avea )
#         temp = "{0:0" + str(theta) + "b}"
#         # print(temp)
#         for index, (attribute, exists) in enumerate(self.attribute_list):
#             sign = -1
#             incrementer = 1
#             if exists:
#                 bin_attr = numpy.array(list(temp.format(attribute)), dtype=int)
#                 e_1 = numpy.add(e_1,  incrementer*bin_attr)
#                 incrementer += 1
#                 span_program_matrix_L[attribute] = bin_attr
#         span_program_matrix_L[0] = e_1
#         return span_program_matrix_L






