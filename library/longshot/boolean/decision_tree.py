from binarytree import Node
from typing import Iterable
from ..error import LongshotError

class DecisionTree:
    """
    Represents a binary decision tree for a boolean formula.
    """
    def __init__(self, ctree: _CppDecisionTree | None = None, root: Node | None = None):
        """
        Initializes a `DecisionTree`. It can be built from a C++ decision tree object (`_CppDecisionTree`) or a given root `Node`.
        """
        if not isinstance(ctree, _CppDecisionTree):
            raise LongshotError("the argument `ctree` is not a DecisionTree object.")
        
        if ctree is not None:
            self._root = self._recursive_build(ctree)
            ctree.delete()
        else:
            self._root = root
        
    def _recursive_build(self, ctree: _CppDecisionTree) -> None:
        """
        Recursively builds the decision tree.
        """
        if ctree.is_constant:
            return Node('T' if bool(ctree.var) else 'F')

        node = Node(ctree.var)        
        node.left = self._recursive_build(ctree.lt)
        node.right = self._recursive_build(ctree.rt)
        
        return node
        
    def decide(self, x: Iterable[int | bool]) -> bool:
        """
        Evaluates the decision tree for a given input assignment `x` and returns the boolean result.
        """
        if not isinstance(x, (int, Iterable)):
            raise LongshotError("the argument `x` is neither an integer nor an iterable.")
        if isinstance(x, int):
            x = [bool((x >> i) & 1) for i in range(MAX_NUM_VARS)]
        
        node = self._root
        
        while node.left is not None and node.right is not None:
            node = node.right if x[node.value] else node.left
        
        return node.value == 'T'
    
    @property
    def root(self) -> Node:
        """The root node of this decision tree."""
        return self._root

    @root.setter
    def root(self, new_root: Node) -> None:
        if not isinstance(new_root, Node):
            raise LongshotError("root must be set to a Node instance")
        self._root = new_root
        