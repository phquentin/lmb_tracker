import unittest
import numpy as np
from lmb.parameters import TrackerParameters
from lmb.gibbs_sampler import gibbs_sampler
from scipy.special import logsumexp

class TestGM(unittest.TestCase):
    def setUp(self):
        self.params = TrackerParameters()
        self.C: np.ndarray = np.asarray([[1,1,9,1,2,2],
                                         [6,2,2,1,1,2],
                                         [1,4,3,9,1,2]], dtype = 'f4') * -1
        self.hyp_weights: np.ndarray = np.zeros(self.C.shape, dtype = 'f4' )
        self.most_lik_assignment = [2,0,3]

    def test_gibbs_sampler(self):
        # Execute gibbs sampler
        gibbs_sampler(self.C, self.hyp_weights, self.params.num_samples, self.params.max_invalid_samples)
        
        # Test whether hypothesis weights sum up to 1 (0 for log)
        self.assertTrue(np.allclose(logsumexp(self.hyp_weights, axis = -1), np.asarray([0, 0, 0]), rtol=1e-05, atol=1e-05))
        
        # Test whether the hypothesis weight matrix has the highest weights for the most likley assignment
        most_lik_assignment = []
        for i in range(len(self.hyp_weights)):
            min_value_index = np.argmin(self.hyp_weights[i]) 
            most_lik_assignment.append(min_value_index)
        self.assertTrue(np.allclose(self.most_lik_assignment, most_lik_assignment))

        # Test resulting shapes of hyp_weights
        self.assertEqual(self.hyp_weights.shape, self.C.shape)      
