from anytree import NodeMixin

node_id = 0


class PolicyTree(NodeMixin):

    def __init__(self, name, tag, children, parent):
        global node_id
        self.name = name
        self.id = node_id
        self.tag = tag
        self.children = children
        self.parent = parent
        node_id += 1

    def __str__(self):
        return f"Node(ID:{self.id},Name:{self.name},Tag:{self.tag},Children:{self.children}, Parent{self.parent} )"



