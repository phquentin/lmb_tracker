import unittest

import numpy as np
from scipy.special import logsumexp

from lmb.murty import murty_wrapper

class TestMurty(unittest.TestCase):
    def test_murty(self):
        C = np.asarray([[.1, 1.1, .9, .1, 4.2, 2.1],
                       [.6, .2, .2, .1, 4.2, 3.2],
                       [.7, .4, .3, .1, 4.2, 2.2]], dtype = 'f4')
        hyp_weights = np.zeros((C.shape[0], C.shape[1] - 0), dtype = 'f4')

        murty_wrapper(C, hyp_weights)

        # # Print resulting existence probabilities
        # print('r ', np.sum(hyp_weights[:,:-1], axis=-1))
        # # Print resulting measurement association probabilities
        # print('r_uk ', np.sum(hyp_weights[:,:], axis=0))

        # Test all weights between 0 and 1
        self.assertTrue(np.all(hyp_weights <= 1) and np.all(hyp_weights >= 0))
        # Test whether hypothesis weights of each target sum up to 1
        self.assertTrue(np.allclose(np.sum(hyp_weights, axis = -1), np.ones(C.shape[0]), rtol=1e-05, atol=1e-05))
        # Test measurement association probability between 0 and 1
        r_uk = np.sum(hyp_weights, axis=0)
        self.assertTrue(np.all(r_uk <= 1))
