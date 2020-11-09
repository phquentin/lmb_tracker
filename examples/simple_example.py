""" 
Simple example to demonstrate the basic functionality 
"""
import os
import sys

import lmb

def main():
    # creating new parameter set
    params = lmb.Parameters() # default parameter set
    tracker = lmb.LMB(params=params)
    print(tracker.params)


if __name__ == '__main__':
    main()
