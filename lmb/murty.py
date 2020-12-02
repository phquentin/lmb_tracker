import queue
import lap
import numpy as np
from numpy import logaddexp

LARGE = 10000.0

def murty_wrapper(C, assignment_weights, n_hyp=1000):
    """
    Solve the LMB ranked assignment using Murty's algorithm

    Computes the n_hyp best assignments and their hypothesis weights. 
    This is a wrapper to extend the cost matrix to meet the requirements
    of Murty's algorithm as described and implemented in 
    [Olofsson, J., Veibäck, C., & Hendeby, G. (2017). 
    Sea ice tracking with a spatially indexed labeled multi-Bernoulli filter. 
    In 20th International Conference on Information Fusion (FUSION). Xi’an, China.]

    Parameters
    ----------
    C : np.ndarray
        Cost matrix (shape: num_tracks x (num_meas + 2))
        The last two columns represent the costs for missed detection and death.
    assignment_weights : np.ndarray, implicit return
        Resulting hypothesis weights (shape: num_tracks x (num_meas + 2))
        The last two columns represent the weights for missed detection and death.
        Each row sums up to 1. 
    n_hyp : int, optional
        Maximum number of samples
    """    
    N = C.shape[0]
    M = C.shape[1] - 2
    # Test equal shapes of cost and weight matrix
    assert assignment_weights.shape == C.shape

    # Extend the cost matrix by missed detection and death entries
    # to meet murty requirements
    miss_ext = LARGE * np.ones((N, N))
    death_ext = LARGE * np.ones((N, N))
    np.fill_diagonal(miss_ext, C[:, -2])
    np.fill_diagonal(death_ext, C[:, -1])
    C_ext = np.concatenate((C[:,:-2], miss_ext, death_ext), axis=1)
    # Test shape of extended cost matrix
    assert C_ext.shape[0] == C.shape[0]
    assert C_ext.shape[1] == (C.shape[1] - 2 + 2*C.shape[0])

    nhyps = 0
    w_sum = 0

    for cost, assignment in murty(C_ext):
        nhyps += 1
        assignment = np.array(assignment)
        w = np.exp(-cost)
        w_sum += w
        ind = assignment < M + 2 * N
        # Rewrite the indices of missed and death to the shape of the weight matrix
        for idx, assign in enumerate(assignment):
            if assign >= M + N:
                # set death weights to last column of weight matrix
                assignment[idx] = M + 1
            elif assign >= M:
                # set missed weights to second last column of weight matrix
                assignment[idx] = M
        assignment_weights[ind, assignment[ind]] += w
        
        if w / w_sum < 1e-4 or nhyps >= n_hyp:
            break

    print("nhyps:", nhyps, "tracks:", N, "meas:", M)

    assignment_weights /= w_sum


def murty(C):
    """Murty algorithm implemented as generator
    
    Parameters
    ----------
    C : np.ndarray
        Cost matrix

    Returns
    -------
    out: (float, list[int])
        Tuple of assignment cost and list of assignment indices
    """
    try:
        Q = queue.PriorityQueue()
        M = C.shape[0]
        N = C.shape[1]
        cost, assign = lap.lapjv(C, extend_cost=True)[0:2]
        Q.put((cost, list(assign),
               (), (),
               (), ()))
        k = 0
        while not Q.empty():
            S = Q.get_nowait()
            yield (S[0], S[1][:M])
            k += 1
            ni = len(S[2])

            rmap = tuple(x for x in range(M) if x not in S[2])
            cmap = tuple(x for x in S[1] if x not in S[3])
            cmap += tuple(x for x in range(N)
                          if x not in S[3] and x not in S[1])

            removed_values = C[S[4], S[5]]
            C[S[4], S[5]] = LARGE

            C_ = C[rmap, :][:, cmap]
            for t in range(M - ni):
                removed_value = C_[t, t]
                C_[t, t] = LARGE

                cost, lassign = lap.lapjv(C_[t:, t:], extend_cost=True)[0:2]
                if LARGE not in C_[range(t, t + len(lassign)), lassign + t]:
                    cost += C[S[2], S[3]].sum()
                    cost += C_[range(t), range(t)].sum()
                    assign = [None] * M
                    for r in range(ni):
                        assign[S[2][r]] = S[3][r]
                    for r in range(t):
                        assign[rmap[r]] = cmap[r]
                    for r in range(len(lassign)):
                        assign[rmap[r + t]] = cmap[lassign[r] + t]

                    nxt = (cost, assign,
                           S[2] + tuple(rmap[x] for x in range(t)),
                           S[3] + tuple(cmap[:t]),
                           S[4] + (rmap[t],),
                           S[5] + (cmap[t],))
                    Q.put(nxt)
                C_[t, t] = removed_value
            C[S[4], S[5]] = removed_values
    except GeneratorExit:
        pass
