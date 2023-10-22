#!/usr/bin/env python3
"""
small test for binary search tree
"""
from engine.tree import BinarySearchTree

def displayTree(root):
    if root:
        displayTree(root.left)
        print(root)
        displayTree(root.right)

def main():

    r = BinarySearchTree(4, 8, 'c')

    assert(not r.isAvailable(2, 5))
    assert(r.isAvailable(0, 3))
    r.insert(0, 3, 'a')
    assert(not r.isAvailable(2, 3))
    assert(r.isAvailable(3, 4))
    r.insert(3, 4, 'b')
    assert(r.isAvailable(12, 15))
    r.insert(12, 15, 'd')
    assert(not r.isAvailable(12, 15))
    try:
        r.insert(13, 11, 'i')
    except AssertionError:
        print("assert(minVal < maxVal) catched")

    displayTree(r)

if __name__ == "__main__":
    main()
