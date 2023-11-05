#!/usr/bin/env python3

import sys
import random as rd
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("./gen_map.py N")
    
    N = int(sys.argv[1])
    print(N)
    nodes = [i for i in range(N)]
    # paths = {i:[] for i in range(N)}

    for n in nodes:
        for i in rd.sample(nodes, rd.randint(2, 4)):
            print(n, i, rd.random()*100+50)

