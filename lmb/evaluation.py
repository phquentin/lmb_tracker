import numpy as np

def create_target_tracks(params=None):
    """
    Creates target tracks with specified simulation length, inital states, motion model, birth and death timestamps 

    Parameters
    ----------
    params: Instance of the Sim_Parameters class

    Returns
    -------
    out: track array of datatype dt_tracks: np.dtype = np.dtype([('x', 'f8',(dim_x+dim_x_dot)),
                                                                 ('ts', 'u4'),
                                                                 ('l', 'u4')])
            
    """
    params = params if params else Sim_Parameters()
                      
    tracks = np.zeros(0, dtype=params.dt_tracks) 

    for ts in range(params.sim_length):

        # predict current targets and append to tracks
        mask_targets_to_predict = tracks['ts']== ts-1
        targets_to_predict = tracks[mask_targets_to_predict]

        # exclude targets from prediction which die at current ts, equals their death
        if len(params.init_track__info[params.init_track__info['death_ts'] == ts]['l']) != 0:
            death_labels = params.init_track__info[params.init_track__info['death_ts'] == ts]['l']
            for l in death_labels:
                targets_to_predict = targets_to_predict[targets_to_predict['l'] != l]

        # create empty predicted_targets 
        predicted_targets = np.zeros(len(targets_to_predict),dtype=params.dt_tracks)

        # -> x = (Fx')' -> x = xF'
        predicted_targets['x'] = np.dot(targets_to_predict['x'],params.F.T)
        predicted_targets['ts'] = ts
        predicted_targets['l'] = targets_to_predict['l']

        tracks = np.concatenate((tracks, predicted_targets))

        # birth of new target at ts

        # extract current birth info for ts
        birth_mask = params.init_track__info['birth_ts']== ts
        ts_init_track_info = params.init_track__info[birth_mask]

        # create empty birth tracks for initialization
        birth_tracks = np.zeros(len(ts_init_track_info),dtype=params.dt_tracks)

        # initialize birth tracks
        for target_i in range(len(ts_init_track_info )):
            birth_tracks['x'][target_i] = ts_init_track_info['x'][target_i]
            birth_tracks['ts'][target_i] = ts
            birth_tracks['l'][target_i] = ts_init_track_info['l'][target_i]

        tracks = np.concatenate((tracks, birth_tracks))

    return tracks

def create_measurement_history(gt_target_track_history, params=None):
    """
    Adds measurement noise to gt_target_tracks 
    TO DO: Reduces gt_target_tracks by missed detections and clutter
    Parameters
    ----------
    gt_target_track_history: track array of data type dt_tracks: np.dtype = np.dtype([('x', 'f8',(dim_x+dim_x_dot)),
                                                                                     ('ts', 'u4'),
                                                                                     ('l', 'u4')])

    params: Instance of the Sim_Parameters class

    Returns
    -------
    measuerement_history: measurement array of data type np.dtype = np.dtype([('z', 'f8',(dim_x)),
                                                                              ('ts', 'u4')])

    """

    params = params if params else Sim_Parameters()

    measuerement_history = np.zeros(len(gt_target_track_history), dtype=params.dt_measuerement)
    measuerement_history['z'] = gt_target_track_history['x'][:,0:params.dim_x]
    measuerement_history['ts'] = gt_target_track_history['ts']
    measuerement_history['z'] += np.random.multivariate_normal(mean = np.zeros(params.dim_x), \
                                                          cov  = params.sigma ** 2 * np.identity(params.dim_x), \
                                                         size  = len(gt_target_track_history['x']))

    return measuerement_history


def evaluate(gt_target_track_history,tracker_estimates_history):
    """
    Evaluates tracker results by commen MOT metrics
    
    Parameters
    ----------
    gt_target_track_history: array_like labeled groundtruth track history

    tracker_estimates_history: array_like labeled track history
    """
    pass
