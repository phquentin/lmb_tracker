""" 
Simple example to demonstrate the basic functionality 
"""
import os
import sys

import lmb

def main():
    # creating new parameter set
    params = lmb.TrackerParameters() # default parameter set
    tracker = lmb.LMB(params=params)
    print(tracker.params)
    print(tracker.targets[0])
    tracker.predict()
    print(tracker.targets[0])


if __name__ == '__main__':
    main()
