#!/usr/bin/env python3
"""
small test for binary search tree
"""
from engine.tree import BinarySearchTree, Nodable

class ImplemNodable(Nodable):

    minVal: int
    maxVal: int
    val: object
    def __init__(self, minVal, maxVal, val) -> None:
        self.minVal = minVal
        self.maxVal = maxVal
        self.val = val

    def minValue(self):
        return self.minVal
    def maxValue(self):
        return self.maxVal
    def __str__(self) -> str:
        return str(self.val)
def main():

    bst = BinarySearchTree()
    bst.insert(ImplemNodable(4, 8, 'c'))
    assert(not bst.insert(ImplemNodable(2, 5, 'i')))
    assert(bst.insert(ImplemNodable(0, 3, 'a')))
    
    assert(not bst.insert(ImplemNodable(2, 3, 'i')))
    assert(bst.insert(ImplemNodable(3, 4, 'b')))
    assert(bst.insert(ImplemNodable(12, 15, 'd')))
    assert(not bst.insert(ImplemNodable(12, 15, 'i')))
    try:
        bst.insert(ImplemNodable(13, 11, 'i'))
    except AssertionError:
        print("assert(minVal < maxVal) catched")

    bst.printTree()

    assert(not bst.search(2, 8))
    assert(bst.search(12, 15) == bst.root.right)
    assert(bst.search(3, 4) == bst.root.left.right)

    print("iterate")
    msg = ""
    for n in bst:
        msg += str(n)
    assert(msg == "abcd")

    print("removing root...")
    n = bst.root.right
    bst.remove(bst.root)
    assert(bst.root == n)
    bst.printTree()
    print("iterate")
    msg = ""
    for n in bst:
        msg += str(n)
    assert(msg == "abd")

if __name__ == "__main__":
    main()
