import motmetrics as mm
import numpy as np


def evaluate_point_2D(tracks_gt, tracks_est, max_d2):
    """
    Evaluates tracker results by MOT Challenge metrics for 2D point tracking problems
    
    Parameters
    ----------
    tracks_gt: ndarray
        Array of ground truth tracks (dtype = SimParameters.dt_tracks)

    tracks_est: ndarray
        Array of estimated tracks (dtype = SimParameters.dt_tracks)

    max_d2: int 
        Maximum squared euclidian distance for which py-motmetrics creates a hypothesis between a ground truth track and estimated track   
    """

    acc = mm.MOTAccumulator(auto_id=True)

    number_ts = np.amax(tracks_gt['ts']) 
   
    for ts in range(number_ts):

        gt_labels_ts = tracks_gt[tracks_gt['ts']==ts]['label']
        est_labels_ts = tracks_est[tracks_est['ts']==ts]['label']
        
        gt_states_ts = tracks_gt[tracks_gt['ts']==ts]['x']
        est_states_ts = tracks_est[tracks_est['ts']==ts]['x']
       
        hypothesis_distance_ts = mm.distances.norm2squared_matrix(gt_states_ts, est_states_ts, max_d2=max_d2)
        
        acc.update(gt_labels_ts, est_labels_ts, hypothesis_distance_ts)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)
