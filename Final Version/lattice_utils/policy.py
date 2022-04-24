import numpy as np
from anytree import Node, AnyNode, RenderTree, AsciiStyle, LevelOrderGroupIter


class Policy:
    def __init__(self, attribute_tree, all_attribute_bound, key_attribute_bound):
        self.attribute_tree = attribute_tree
        self.all_attribute_bound = all_attribute_bound
        self.key_attribute_bound = key_attribute_bound
        self.lsss_matrix = self.lewko_waters_algorithm()
        self.attribute_length = len(self.lsss_matrix[0])
        self.attribute_limit = len(self.lsss_matrix)

    def __repr__(self):
        return self.attribute_tree.__repr__

    def __str__(self):
        for pre, fill, node in RenderTree(self.attribute_tree):
            print("%s%s %s" % (pre, node.name, node.tag))
        print(f"Policy Tree Matrix:\n {np.matrix(self.lsss_matrix)}")
        return "Finished Tree and Linear Secret Sharing Matrix"

    def return_theta(self):
        return self.attribute_length

    def return_attribute_limit(self):
        return self.attribute_limit

    def lewko_waters_algorithm(self):
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
        return lsss_matrix








