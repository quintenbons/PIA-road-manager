from __future__ import annotations

class BinarySearchTree:
    left: BinarySearchTree
    right: BinarySearchTree

    minValue: float
    maxValue: float
    obj: None

    def __init__(self, minVal, maxVal, obj = None, left = None, right = None) -> None:
        assert(minVal < maxVal)
        self.minValue = minVal
        self.maxValue = maxVal
        self.left = left
        self.right = right
        self.obj = obj
    
    def __str__(self) -> str:
        return f"{(self.minValue, self.maxValue, self.obj)}"
    
    def insert(self, minVal, maxVal, obj):
        if maxVal <= self.minValue:
            if self.left is None:
                self.left = BinarySearchTree(minVal, maxVal, obj)
            else:
                self.left.insert(minVal, maxVal, obj)
        elif minVal >= self.maxValue:
            if self.right is None:
                self.right = BinarySearchTree(minVal, maxVal, obj)
            else:
                self.right.insert(minVal, maxVal, obj)

    def isAvailable(self, minVal, maxVal):
        if maxVal <= self.minValue:
            return True if self.left is None else self.left.isAvailable(minVal, maxVal)
        elif minVal >= self.maxValue:
            return True if self.right is None else self.right.isAvailable(minVal, maxVal)
        return False