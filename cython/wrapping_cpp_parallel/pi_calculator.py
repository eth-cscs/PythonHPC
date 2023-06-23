#!/usr/bin/python3

import sys
from pi_mc import pi_mc


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <int n>')
        exit(-1)

    pi = pi_mc(int(sys.argv[1]))
    print(f'PI is: {pi:.10f}')
