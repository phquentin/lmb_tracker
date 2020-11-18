import lapjv
import numpy as np
from numpy import logaddexp

def murty(C, assignment_weights, n_hyp=100):
    """
    Solving ranked assignment using Murty's algorithm

    Computes the n_hyp best assignments and their hypothesis weights

    Parameters
    ----------
    C : np.ndarray
        Cost matrix
    assignment_weights : np.ndarray, implicit return
        Resulting hypothesis weights, same shape as C
    n_hyp : int, optional
        Maximum number of samples
    """
    cost, assignment = lapjv.lap(C, extend_cost=True)[0:2]
    # print('cost=',cost)
    # print(assignment)
    ind = assignment < assignment_weights.shape[0] + assignment_weights.shape[1] - 1
    assignment[assignment >= assignment_weights.shape[1] - 1] = assignment_weights.shape[1] - 1
    assignment_weights[ind, assignment[ind]] = logaddexp(assignment_weights[ind, assignment[ind]], -cost)
