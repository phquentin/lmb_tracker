import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from .murty import murty_wrapper
from .gibbs_sampler import gibbs_sampler

float_precision = 'f4'

@dataclass (frozen=True)
class TrackerParameters():
    """
    Class containing the overall tracker parameters
    """
    dim_x: int = 4              # Dimension (number) of states
    dim_z: int = 2              # Dimension (number) of measurement inputs
    n_targets_max: int = 1000   # maximum number of targets
    n_gm_cmpnts_max: int = 100  # maximum number of Gaussian mixture components
    log_w_prun_th: float = np.log(0.2)       # Log-likelihood threshold of gaussian mixture weight for pruning 
    log_r_sel_th: float = np.log(0.2) # Log-likelihood threshold of target existence probability for selection
    p_survival: float = 0.99    # survival probability
    p_birth: float = 0.2        # birth probability
    adaptive_birth_th: float = 1e-3 # adaptive birth threshold
    p_detect: float = 0.99      # detection probability
    log_p_detect: float = field(init=False)
    log_q_detect: float = field(init=False)
    kappa: float = 0.01         # clutter intensity
    log_kappa: float = field(init=False)
    r_prun_th: float = 0.05    # existence probability pruning threshold
    log_r_prun_th: float = field(init=False)
    # observation noise covariance
    R: np.ndarray = np.asarray([[2., 0.],
                                [0., 2.]], dtype=float_precision)
    # process noise covariance
    Q: np.ndarray = np.asarray([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [1., 0., 1., 0.],
                                [0., 1., 0., 1.]], dtype=float_precision)
    # Motion model: state transition matrix
    F: np.ndarray = np.asarray([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]], dtype=float_precision)
    # Observation model
    H: np.ndarray = np.asarray([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], dtype=float_precision)
    # Initial state covariance matrix
    P_init: np.ndarray = np.asarray([[2., 0., 0., 0.],
                                    [0., 2., 0., 0.],
                                    [0., 0., 2., 0.],
                                    [0., 0., 0., 2.]], dtype=float_precision)
    # Algorithm used for solving the ranked assignment problem
    ranked_assign: Callable[[np.ndarray, np.ndarray, int], None] = murty_wrapper
    num_assignments: int = 1000 # Maximum number of hypothetical assignments created by the ranked assignment

    # Gibbs sampler parameters
    num_samples: int = 1000  # Number of samples the Gibbs sampler takes from the eta_nll matrix
    max_invalid_samples: int = 100 # Maximum number of consecutive invalid samples that do not contain a valid assignment after that the gibbs sampler terminates

    def __post_init__(self):
        """
        Initialization of computed attributes
        """
        object.__setattr__(self, 'log_p_detect', np.log(self.p_detect))
        object.__setattr__(self, 'log_q_detect', np.log(1 - self.p_detect))
        object.__setattr__(self, 'log_kappa', np.log(self.kappa))
        object.__setattr__(self, 'log_r_prun_th', np.log(self.r_prun_th))

@dataclass (frozen=True) 
class SimParameters():
    """
    Class containing the overall simulation parameters
    """                      
    sim_length: int = 9  # number of simulation timesteps
    dim_x: int = 4 # Dimension (number) of state variables
    dim_z: int = 2 # Dimension of measured state variables
    sigma: float = 0 # Standard deviation of measurement noise
    max_d2: int = 1000**2 # Maximum squared euclidian distance for which py-motmetrics creates a hypothesis between a ground truth track and estimated track

    # State Transition matrix
    F: np.ndarray = np.asarray([[1,0,1,0],
                                [0,1,0,1],
                                [0,0,1,0],
                                [0,0,0,1]],dtype=float_precision)

    # Data type of array to generate tracks
    dt_init_track_info: np.dtype = np.dtype([('x', 'f8',(dim_x)),
                                     ('birth_ts', 'u4'),
                                     ('death_ts', 'u4'),
                                     ('label', 'f4')])
    # Data type of tracks
    dt_tracks: np.dtype = np.dtype([('x', 'f8',(dim_x)),
                          ('ts', 'u4'),
                          ('label', 'f4'),
                          ('r', 'f4')])

    # Data type of measuerements
    dt_measurement: np.dtype = np.dtype([('z', 'f8',(dim_z)),
                                          ('ts', 'u4')])

    # Array with state, birth and death information to generate tracks
    init_track_info: np.ndarray = np.asarray([([10, 10, 2, 2],0, 7, 1.0),
                                              ([20, 50, 4, 5],0, 21, 2.0),
                                              ([35, 40, -3, -4],0, 21, 3.0),
                                              ([30, 90, 2, -4],0, 21, 4.0)],dtype=dt_init_track_info)

