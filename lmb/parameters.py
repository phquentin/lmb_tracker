import numpy as np
from dataclasses import dataclass, field

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
    log_p_detect: float = field(init=False)
    log_q_detect: float = field(init=False)
    kappa: float = 0.01         # clutter intensity
    log_kappa: float = field(init=False)
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

    
    def __post_init__(self):
        """
        Initialization of computed attributes
        """
        object.__setattr__(self, 'log_p_detect', np.log(self.p_detect))
        object.__setattr__(self, 'log_q_detect', np.log(1 - self.p_detect))
        object.__setattr__(self, 'log_kappa', np.log(self.kappa))
