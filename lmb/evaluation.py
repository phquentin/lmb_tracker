import motmetrics as mm
import numpy as np
from .parameters import SimParameters

def evaluate(gt_target_track_history, tracker_estimates_history, params=None):
    """
    Evaluates tracker results by MOT Challenge metrics
    
    Parameters
    ----------
    gt_target_track_history: ndarray
        Array of ground truth tracks (dtype = SimParameters.dt_tracks)

    tracker_estimates_history: ndarray
        Array of estimated tracks (dtype = SimParameters.dt_tracks)

    params: Instance of the SimParameters class    
    """

    params = params if params else SimParameters()
    acc = mm.MOTAccumulator(auto_id=True)

    for ts in range(params.sim_length):

        gt_labels_ts = gt_target_track_history[gt_target_track_history['ts']==ts]['label']
        track_labels_ts = tracker_estimates_history[tracker_estimates_history['ts']==ts]['label']
        
        gt_states_ts = gt_target_track_history[gt_target_track_history['ts']==ts]['x']
        track_states_ts = tracker_estimates_history[tracker_estimates_history['ts']==ts]['x']
       
        hypothesis_distance_ts = mm.distances.norm2squared_matrix(gt_states_ts, track_states_ts, max_d2=params.max_d2)
        
        acc.update(gt_labels_ts,track_labels_ts,hypothesis_distance_ts)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)




