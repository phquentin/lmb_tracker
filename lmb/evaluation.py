
def create_target_tracks(params=None):
    """
    Creates target tracks with specified time_frame, inital states, motion model, birth and death timestamps 
    """
    _params = params if params else Sim_Parameters()
    pass


def create_measurement_history(gt_target_track_history, params=None):
    """
    Reduces gt_target_tracks by missed detections, adds measurement noise and clutter
    """
    _params = params if params else Sim_Parameters()
    pass


def evaluate(gt_target_track_history,tracker_estimates_history):
    """
    Evaluates tracker results by commen MOT metrics
    """
    pass
