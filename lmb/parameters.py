import numpy as np
from dataclasses import dataclass

@dataclass (frozen=True)
class Parameters():
    """
    Class containing the overal tracker parameters
    """
    dim_x: int = 4              # Dimension (number) of states (first + second order)
    dim_z: int = 2              # Dimension (number) of measurement inputs
    n_targets_max: int = 1000   # maximum number of targets
    n_gm_cmpnts_max: int = 100  # maximum number of Gaussian mixture components
    p_survival: float = 0.99    # survival probability
    p_birth: float = 0.02       # birth probability
    p_detect: float = 0.99      # detection probability
    kappa: float = 0.01         # clutter intensity
    r_prun_th: float = 1e-3     # existence probability pruning threshold
    # observation noise covariance
    R: np.ndarray = np.asarray([[10., 0.],
                                [0., 10.]], dtype='f4')
    # process noise covariance
    Q: np.ndarray = np.asarray([[5., 0., 10., 0.],
                                [0., 5., 0., 10.],
                                [10., 0., 20., 0.],
                                [0., 10., 0., 20.]], dtype='f4')
    # Motion model: state transition matrix
    F: np.ndarray = np.asarray([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]], dtype='f4')
    # Observation model
    H: np.ndarray = np.asarray([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], dtype='f4')
    # Initial state covariance matrix
    P_init: np.ndarray = np.asarray([[100., 0., 0., 0.],
                                    [0., 100., 0., 0.],
                                    [0., 0., 100., 0.],
                                    [0., 0., 0., 100.]], dtype='f4')
