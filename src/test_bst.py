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

    r = BinarySearchTree()
    # r.insert(4, 8, 'c')
    r.insert(ImplemNodable(4, 8, 'c'))
    assert(not r.insert(ImplemNodable(2, 5, 'i')))
    assert(r.insert(ImplemNodable(0, 3, 'a')))
    
    assert(not r.insert(ImplemNodable(2, 3, 'i')))
    assert(r.insert(ImplemNodable(3, 4, 'b')))
    assert(r.insert(ImplemNodable(12, 15, 'd')))
    assert(not r.insert(ImplemNodable(12, 15, 'i')))
    try:
        r.insert(ImplemNodable(13, 11, 'i'))
    except AssertionError:
        print("assert(minVal < maxVal) catched")

    r.printTree()

    assert(not r.search(2, 8))
    assert(r.search(12, 15) == r.root.right)
    assert(r.search(3, 4) == r.root.left.right)

    print("iterate")
    msg = ""
    for n in r.iterate():
        msg += str(n)
    assert(msg == "abcd")

    print("removing root...")
    n = r.root.right
    r.remove(r.root)
    assert(r.root == n)
    r.printTree()
    print("iterate")
    msg = ""
    for n in r.iterate():
        msg += str(n)
    assert(msg == "abd")
if __name__ == "__main__":
    main()
