class Tree:
    def __init__(self, leafnode_count, splitnode_count):
        self.leafnode_count = leafnode_count  # number of leaf nodes in a single tree
        self.splitnode_count = splitnode_count # number of split nodes in a single tree

class LeafNode:
    def __init__(self, value):
        self.value = value # output of the leaf node
        self.lb = [] # lowerbounds of the features
        self.ub = [] # upperbounds of the features
        self.splitnode_ids = [] # split nodes on the route to the leaf node
        self.splitrule_sign = [] # sign of the rule on the split nodes, 0 if <= (true), 1 if > (false)

class SplitNode:
    def __init__(self, variable, criterion, leftchild_id, leftchild_is_leaf, rightchild_id, rightchild_is_leaf):
        self.variable = variable # variable used in the split
        self.criterion = criterion # split criterion
        self.leftchild_id = leftchild_id # id of the split node's left child, can be leaf node id or split node id
        self.leftchild_is_leaf = leftchild_is_leaf # 1 if leaf node, then the above id is leaf id; 0 if split node, then the id is split node id
        self.rightchild_id = rightchild_id # id of the split node's left child, can be leaf node id or split node id
        self.rightchild_is_leaf = rightchild_is_leaf # 1 if leaf node, then the above id is leaf id; 0 if split node, then the id is split node id
        self.splitnode_ids = [] # split nodes on the route to the split node
        self.splitrule_signs = [] # sign of the rule on the split nodes, 0 if <= (true), 1 if > (false)
        