import numpy as np
from dataclasses import dataclass

@dataclass (frozen=True)
class Tracker_Parameters():
    """
    Class containing the overal tracker parameters
    """
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
    Q: np.ndarray = np.asarray([[5, 0, 10, 0],
                                [0, 5, 0, 10],
                                [10, 0, 20, 0],
                                [0, 10, 0, 20]], dtype='f4')

@dataclass (frozen=True) 
class Sim_Parameters():
    """
    Class containing the overal simulation parameters
    """                      
    sim_length: int = 4   # number of simulation timesteps
    dim_x: int = 2 # Dimension (number) of position state variables
    dim_x_dot: int = 2 # Dimension (number) of the velocity state variables
    sigma: float = 2 # Standard deviation of measurement noise

    # State Transition matrix
    F: np.ndarray = np.asarray([[1,0,1,0],
                                [0,1,0,1],
                                [0,0,1,0],
                                [0,0,0,1]],dtype='f4')

    # Data type of array to generate tracks
    dt_init_track_info: np.dtype = np.dtype([('x', 'f8',(dim_x+dim_x_dot)),
                                     ('birth_ts', 'u4'),
                                     ('death_ts', 'u4'),
                                     ('l', 'u4')])
    # Data type of tracks
    dt_tracks: np.dtype = np.dtype([('x', 'f8',(dim_x+dim_x_dot)),
                          ('ts', 'u4'),
                          ('l', 'u4')])

    # Data type of measuerements
    dt_measuerement: np.dtype = np.dtype([('z', 'f8',(dim_x)),
                                          ('ts', 'u4')])

    # Array with state, birth and death information to generate tracks
    init_track__info: np.ndarray = np.asarray ([([10, 10, 2, 2],0, 2, 1),
                                                ([20, 50, 4, 5],0, 3, 2), 
                                                ([20, 50, 4, 5],2, 5, 3)],dtype=dt_init_track_info)
