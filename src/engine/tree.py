from __future__ import annotations
from typing import TypeVar

class BinarySearchTree:
    root: Node
    
    def __init__(self):
        self.root = None

    def insertCmp(self, cmpMin, cmpMax, obj) -> bool:
        if self.root is None:
            self.root = Node(obj)
            return True
        if self.root.isAvailable(cmpMin, cmpMax):
            self.root.insert(obj)
            return True
        return False
    #Duplicate : performance issue
    def insert(self, obj: Nodable) -> bool:
        if self.root is None:
            self.root = Node(obj)
            return True
        if self.root.isAvailable(obj.minValue(), obj.maxValue()):
            self.root.insert(obj)
            return True
        return False

    def search(self, minVal, maxVal) -> Node:
        current = self.root
        #TODO optimize with precondition after
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

    def iterate(self):
        if self.root:
            yield from self.root.iterate()

    def remove(self, node: Node):
        if node.left is None:
            self.shiftNode(node, node.right)
        elif node.right is None:
            self.shiftNode(node, node.left)
        else:
            e = node.successor()
            if e.parent is not node:
                self.shiftNode(e, e.right)
                e.right = node.right
                e.right.parent = e
            self.shiftNode(node, e)
            e.left = node.left
            e.left.parent = e

    def shiftNode(self, node1: Node, node2: Node):
        if node1.parent is None:
            self.root = node2
        elif node1 == node1.parent.left:
            node1.parent.left = node2
        else:
            node1.parent.right = node2
        if node2 is not None:
            node2.parent = node1.parent

class Node:
    left: Node
    right: Node
    parent: Node
    obj: Nodable

    def __init__(self, obj: Nodable, parent: Node = None, left = None, right = None) -> None:
        assert(obj.minValue() < obj.maxValue())

        self.obj = obj
        self.left = left
        self.right = right
        self.parent = parent
        self.obj.bindTree(self)
    
    def __str__(self) -> str:
        return f"{(self.obj.minValue(), self.obj.maxValue(), str(self.obj))}"
    
    def insert(self, obj: Nodable):
        if obj.maxValue() <= self.obj.minValue():
            if self.left is None:
                self.left = Node(obj, self)
            else:
                self.left.insert(obj)
        elif obj.minValue() >= self.obj.maxValue():
            if self.right is None:
                self.right = Node(obj, self)
            else:
                self.right.insert(obj)

    def isAvailable(self, minVal, maxVal) -> bool:
        if maxVal <= self.obj.minValue():
            return True if self.left is None else self.left.isAvailable(minVal, maxVal)
        elif minVal >= self.obj.maxValue():
            return True if self.right is None else self.right.isAvailable(minVal, maxVal)
        return False

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
        pass
    def minValue(self):
        pass

    def bindTree(self, node: Node):
        pass