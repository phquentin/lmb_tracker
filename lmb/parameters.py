import numpy as np
from dataclasses import dataclass, field

@dataclass (frozen=True)
class TrackerParameters():
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

@dataclass (frozen=True) 
class SimParameters():
    """
    Class containing the overal simulation parameters
    """                      
    sim_length: int = 4   # number of simulation timesteps
    dim_x: int = 4 # Dimension (number) of state variables
    dim_z: int = 2 # Dimension of measured state variables
    sigma: float = 2 # Standard deviation of measurement noise

    # State Transition matrix
    F: np.ndarray = np.asarray([[1,0,1,0],
                                [0,1,0,1],
                                [0,0,1,0],
                                [0,0,0,1]],dtype='f4')

    # Data type of array to generate tracks
    dt_init_track_info: np.dtype = np.dtype([('x', 'f8',(dim_x)),
                                     ('birth_ts', 'u4'),
                                     ('death_ts', 'u4'),
                                     ('label', 'u4')])
    # Data type of tracks
    dt_tracks: np.dtype = np.dtype([('x', 'f8',(dim_x)),
                          ('ts', 'u4'),
                          ('label', 'u4')])

    # Data type of measuerements
    dt_measurement: np.dtype = np.dtype([('z', 'f8',(dim_z)),
                                          ('ts', 'u4')])

    # Array with state, birth and death information to generate tracks
    init_track_info: np.ndarray = np.asarray ([([10, 10, 2, 2],0, 2, 1),
                                                ([20, 50, 4, 5],0, 3, 2), 
                                                ([20, 50, 4, 5],2, 5, 3)],dtype=dt_init_track_info)
