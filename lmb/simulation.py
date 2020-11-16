import numpy as np
from .parameters import SimParameters

def create_target_tracks(params=None):
    """
    Creates target tracks with specified simulation length, inital states, motion model, birth and death timestamps 

    Parameters
    ----------
    params: Instance of the SimParameters class

    Returns
    -------
    out: ndarray
        Array of Tracks (dtype = np.dtype([('x', 'f8',(dim_x+dim_x_dot)),
                                           ('ts', 'u4'),
                                           ('l', 'u4')]))
            
    """
    params = params if params else SimParameters()
                      
    tracks = np.zeros(0, dtype=params.dt_tracks) 

    for ts in range(params.sim_length):

        # predict current targets and append to tracks
        mask_targets_to_predict = tracks['ts']== ts-1
        targets_to_predict = tracks[mask_targets_to_predict]

        # exclude targets from prediction which die at current ts, equals their death
        death_labels = params.init_track_info[params.init_track_info['death_ts'] == ts]['label']
        for l in death_labels:
            targets_to_predict = targets_to_predict[targets_to_predict['label'] != l]

        # create empty predicted_targets 
        predicted_targets = np.zeros(len(targets_to_predict),dtype=params.dt_tracks)

        # -> x = (Fx')' -> x = xF'
        predicted_targets['x'] = np.dot(targets_to_predict['x'],params.F.T)
        predicted_targets['ts'] = ts
        predicted_targets['label'] = targets_to_predict['label']

        tracks = np.concatenate((tracks, predicted_targets))

        # birth of new target at ts

        # extract current birth info for ts
        birth_mask = params.init_track_info['birth_ts']== ts
        init_track_info_of_ts = params.init_track_info[birth_mask]

        # create empty birth tracks for initialization
        birth_tracks = np.zeros(len(init_track_info_of_ts),dtype=params.dt_tracks)

        # initialize birth tracks
        for i, birth_target in enumerate(init_track_info_of_ts):
            birth_tracks['x'][i] = birth_target['x']
            birth_tracks['ts'][i] = ts
            birth_tracks['label'][i] = birth_target['label']
            
        tracks = np.concatenate((tracks, birth_tracks))

    return tracks

def create_measurement_history(gt_target_track_history, params=None):
    """
    Adds measurement noise to gt_target_tracks 
    TODO: Reduces gt_target_tracks by missed detections and add clutter

    Parameters
    ----------
    gt_target_track_history: ndarray
        Array of ground truth tracks (dtype = np.dtype([('x', 'f8',(dim_x+dim_x_dot)),
                                                       ('ts', 'u4'),
                                                       ('l', 'u4')])))
                                                    
    params: Instance of the SimParameters class

    Returns
    -------
    out: ndarray
        Array of the measurment history (dtype = np.dtype([('z', 'f8',(dim_x)),
                                                           ('ts', 'u4')]))
    """

    params = params if params else SimParameters()

    measurement_history = np.zeros(len(gt_target_track_history), dtype=params.dt_measurement)
    measurement_history['z'] = gt_target_track_history['x'][:,0:params.dim_z]
    measurement_history['ts'] = gt_target_track_history['ts']
    measurement_history['z'] += np.random.multivariate_normal(mean = np.zeros(params.dim_z), \
                                                          cov  = params.sigma ** 2 * np.identity(params.dim_z), \
                                                         size  = len(gt_target_track_history['x']))

    return measurement_history
