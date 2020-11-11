import unittest
from copy import deepcopy
from unittest.mock import MagicMock
import numpy as np

import lmb.gm

class TestGM(unittest.TestCase):
    def setUp(self):
        self.params = MagicMock()

        self.params.dim_x = 4
        # observation noise covariance
        self.params.R: np.ndarray = np.asarray([[10., 0.],
                                        [0., 10.]], dtype='f4')
        # process noise covariance
        self.params.Q: np.ndarray = np.asarray([[5., 0., 10., 0.],
                                [0., 5., 0., 10.],
                                [10., 0., 20., 0.],
                                [0., 10., 0., 20.]], dtype='f4')
        # Motion model: state transition matrix
        self.params.F: np.ndarray = np.asarray([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]], dtype='f4')
        # Observation model
        self.params.H: np.ndarray = np.asarray([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], dtype='f4')
        # Initial state covariance matrix
        self.params.P_init: np.ndarray = np.asarray([[100., 0., 0., 0.],
                                    [0., 100., 0., 0.],
                                    [0., 0., 100., 0.],
                                    [0., 0., 0., 100.]], dtype='f4')
        self.pdf = lmb.GM(self.params)
        self.pdf.mc = np.append(self.pdf.mc[0], self.pdf.mc[0])
        self.pdf.mc[0]['x'] = np.asarray([1., 0., 0.5, 0.5])
        self.pdf.mc[1]['x'] = np.asarray([-5., 0., -1., 2])

    def test_predict(self):
        mc_prior = deepcopy(self.pdf.mc)
        self.pdf.predict()
        # Test states
        self.assertTrue(np.allclose(self.pdf.mc[0]['x'], np.asarray([1.5, 0.5, 0.5, 0.5])))
        self.assertTrue(np.allclose(self.pdf.mc[1]['x'], np.asarray([-6., 2., -1., 2.])))
        # Test log_w
        self.assertTrue(np.allclose(self.pdf.mc['log_w'], mc_prior['log_w']))
        # Test shape of P
        self.assertEqual(self.pdf.mc['P'].shape, mc_prior['P'].shape)
