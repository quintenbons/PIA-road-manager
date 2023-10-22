#!/usr/bin/env python3
"""
small test for binary search tree
"""
from engine.tree import BinarySearchTree


def main():

    r = BinarySearchTree()
    r.insert(4, 8, 'c')

    assert(not r.insert(2, 5, 'i'))
    assert(r.insert(0, 3, 'a'))
    
    assert(not r.insert(2, 3, 'i'))
    assert(r.insert(3, 4, 'b'))
    assert(r.insert(12, 15, 'd'))
    assert(not r.insert(12, 15, 'i'))
    try:
        r.insert(13, 11, 'i')
    except AssertionError:
        print("assert(minVal < maxVal) catched")

    r.printTree()

if __name__ == "__main__":
    main()
