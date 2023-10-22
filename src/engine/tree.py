from __future__ import annotations

class BinarySearchTree:
    root: Node
    
    def __init__(self):
        self.root = None

    def insertCmp(self, minVal, maxVal, cmpMin, cmpMax, obj) -> bool:
        if self.root is None:
            self.root = Node(minVal, maxVal, obj)
            return True
        if self.root.isAvailable(cmpMin, cmpMax):
            self.root.insert(minVal, maxVal, obj)
            return True
        return False
    #Duplicate : performance issue
    def insert(self, minVal, maxVal, obj) -> bool:
        if self.root is None:
            self.root = Node(minVal, maxVal, obj)
            return True
        if self.root.isAvailable(minVal, maxVal):
            self.root.insert(minVal, maxVal, obj)
            return True
        return False

    def search(self, minVal, maxVal) -> Node:
        current = self.root
        #TODO optimize with precondition after
        while current is not None and (current.minValue != minVal and current.maxValue != maxVal):
            if maxVal <= current.minValue:
                current = current.left
            elif minVal >= current.maxValue:
                current = current.right
            else:
                return None
        if current.minValue == minVal and current.maxValue == maxVal:
            return current
        else:
            return None

    def printTree(self):
        if self.root:
            self.root.printTree()

    
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
    # def remove(self, minVal, maxVal):
    #     if self.minValue == minVal and self.maxValue == maxVal:
    #         ...

    # def shiftNodes

class Node:
    left: Node
    right: Node
    parent: Node

    minValue: float
    maxValue: float
    obj: None

    def __init__(self, minVal, maxVal, obj = None, parent: Node = None, left = None, right = None) -> None:
        assert(minVal < maxVal)
        self.minValue = minVal
        self.maxValue = maxVal
        self.obj = obj
        self.left = left
        self.right = right
        self.parent = parent
    
    def __str__(self) -> str:
        return f"{(self.minValue, self.maxValue, self.obj)}"
    
    def insert(self, minVal, maxVal, obj):
        if maxVal <= self.minValue:
            if self.left is None:
                self.left = Node(minVal, maxVal, obj, self)
            else:
                self.left.insert(minVal, maxVal, obj)
        elif minVal >= self.maxValue:
            if self.right is None:
                self.right = Node(minVal, maxVal, obj, self)
            else:
                self.right.insert(minVal, maxVal, obj)

    def isAvailable(self, minVal, maxVal) -> bool:
        if maxVal <= self.minValue:
            return True if self.left is None else self.left.isAvailable(minVal, maxVal)
        elif minVal >= self.maxValue:
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

    def printTree(self):
        if self.left:
            self.left.printTree()
        print(self)
        if self.right:
            self.right.printTree()