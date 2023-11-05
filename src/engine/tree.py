from __future__ import annotations
from typing import TypeVar


class BinarySearchTree:
    root: Node
    count: int

    def __init__(self):
        self.root = None
        self.count = 0

    def insertCmp(self, cmpMin, cmpMax, obj) -> bool:
        if self.root is None:
            self.root = Node(self, obj)
            self.count = 1
            return True
        if self.root.isAvailable(cmpMin, cmpMax):
            self.root.insert(self, obj)
            self.count += 1
            return True
        return False
    # Duplicate : performance issue

    def insert(self, obj: Nodable) -> bool:
        if self.root is None:
            self.root = Node(self, obj)
            self.count = 1
            return True
        if self.root.isAvailable(obj.minValue(), obj.maxValue()):
            self.root.insert(self, obj)
            self.count += 1
            return True
        return False

    def search(self, minVal, maxVal) -> Node:
        current = self.root
        # TODO optimize with precondition
        while current is not None and (current.obj.minValue() != minVal and current.obj.maxValue() != maxVal):
            if maxVal <= current.obj.minValue():
                current = current.left
            elif minVal >= current.obj.maxValue():
                current = current.right
            else:
                return None
        if current.obj.minValue() == minVal and current.obj.maxValue() == maxVal:
            return current
        else:
            return None

    def printTree(self):
        if self.root:
            self.root.printTree()

    def __iter__(self):
        if self.root:
            yield from self.root.iterate()

    def remove(self, node: Node):
        assert (node._bst == self)
        node.remove()
        self.count -= 1
    
    def __len__(self):
        return self.count


class Node:
    left: Node
    right: Node
    parent: Node
    obj: Nodable
    _bst: BinarySearchTree

    def __init__(self, bst: BinarySearchTree, obj: Nodable, parent: Node = None, left=None, right=None) -> None:
        assert (obj.minValue() < obj.maxValue())
        self._bst = bst
        self.obj = obj
        self.left = left
        self.right = right
        self.parent = parent
        self.obj.bindTree(self)

    def __str__(self) -> str:
        return f"{(self.obj.minValue(), self.obj.maxValue(), str(self.obj))}"

    def insert(self, bst: BinarySearchTree, obj: Nodable):
        if obj.maxValue() <= self.obj.minValue():
            if self.left is None:
                self.left = Node(bst, obj, self)
            else:
                self.left.insert(bst, obj)
        elif obj.minValue() >= self.obj.maxValue():
            if self.right is None:
                self.right = Node(bst, obj, self)
            else:
                self.right.insert(bst, obj)

    def isAvailable(self, minVal, maxVal) -> bool:
        if maxVal <= self.obj.minValue():
            return True if self.left is None else self.left.isAvailable(minVal, maxVal)
        elif minVal >= self.obj.maxValue():
            return True if self.right is None else self.right.isAvailable(minVal, maxVal)
        return False

    def remove(self):
        if self.left is None:
            self.shiftNode(self.right)
        elif self.right is None:
            self.shiftNode(self.left)
        else:
            successor = self.successor()
            if successor.parent is not self:
                successor.shiftNode(successor.right)
                successor.right = self.right
                successor.right.parent = successor
            self.shiftNode(successor)
            successor.left = self.left
            successor.left.parent = successor

    def shiftNode(self, other: Node):
        if self.parent is None:
            self._bst.root = other
        elif self == self.parent.left:
            self.parent.left = other
        else:
            self.parent.right = other
        if other is not None:
            other.parent = self.parent

    def successor(self) -> Node:
        if self.right is not None:
            return self.right.minimum()
        x = self
        y = self.parent
        while y is not None and x == y.right():
            x = y
            y = y.parent

        return y

    def predecessor(self) -> Node:
        if self.left is not None:
            return self.left.maximum()
        x = self
        y = self.parent

        while y is not None and x == y.left:
            x = y
            y = y.parent

        return y

    def maximum(self) -> Node:
        x = self
        while x.right is not None:
            x = x.right
        return x

    def minimum(self) -> Node:
        x = self
        while x.left is not None:
            x = x.left
        return x

    def iterate(self) -> object:
        if self.left:
            yield from self.left.iterate()
        yield self.obj
        if self.right:
            yield from self.right.iterate()

    def printTree(self):
        if self.left:
            self.left.printTree()
        print(self)
        if self.right:
            self.right.printTree()


class Nodable:
    """Interface used by this BST"""

    def maxValue(self):
        raise NotImplementedError

    def minValue(self):
        raise NotImplementedError

    def bindTree(self, node: Node):
        raise NotImplementedError

    def getNode(self):
        raise NotImplementedError

