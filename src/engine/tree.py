from __future__ import annotations

class BinarySearchTree:
    root: Node
    
    def __init__(self):
        self.root = None

    def insertCmp(self, minVal, maxVal, cmpMin, cmpMax, obj):
        if self.root is None:
            self.root = Node(minVal, maxVal, obj)
            return True
        if self.root.isAvailable(cmpMin, cmpMax):
            self.root.insert(minVal, maxVal, obj)
            return True
        return False
    #Duplicate : performance issue
    def insert(self, minVal, maxVal, obj):
        if self.root is None:
            self.root = Node(minVal, maxVal, obj)
            return True
        if self.root.isAvailable(minVal, maxVal):
            self.root.insert(minVal, maxVal, obj)
            return True
        return False

    def printTree(self):
        if self.root:
            self.root.printTree()

    
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

    def isAvailable(self, minVal, maxVal):
        if maxVal <= self.minValue:
            return True if self.left is None else self.left.isAvailable(minVal, maxVal)
        elif minVal >= self.maxValue:
            return True if self.right is None else self.right.isAvailable(minVal, maxVal)
        return False

    def printTree(self):
        if self.left:
            self.left.printTree()
        print(self)
        if self.right:
            self.right.printTree()