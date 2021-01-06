import unittest

import numpy as np

from lmb.utils import esf

class TestUtils(unittest.TestCase):
    def test_esf(self):
        z = np.asarray([2., 0.3, 5.0, 11.0])
        cd0 = 1
        cd1 = z[0] + z[1] + z[2] + z[3]
        cd2 = z[0]*z[1] + z[0]*z[2] + z[0]*z[3] + z[1]*z[2] + z[1]*z[3] + z[2]*z[3]
        cd3 = z[0]*z[1]*z[2] + z[0]*z[1]*z[3] + z[0]*z[2]*z[3] + z[1]*z[2]*z[3]
        cd4 = z[0]*z[1]*z[2]*z[3]
        cd_gt = np.asarray([cd0, cd1, cd2, cd3, cd4])
        
        cd_list = esf(z)

        self.assertTrue(np.allclose(cd_gt, cd_list))
