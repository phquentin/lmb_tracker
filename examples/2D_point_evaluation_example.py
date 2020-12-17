""" 
Simple 2D point evaluation example to test and evaluate tracker for basic functionality
No full functionality since the tracker is not fully implemented. Uncomment lmb.evaluate_point_2D, when tracker is 
implemented.
"""

import lmb
import numpy as np

def main():
    
    sim_params = lmb.SimParameters() 
    tracker_params = lmb.TrackerParameters()

    tracker = lmb.LMB(params=tracker_params)

    gt_target_track_history = lmb.create_target_tracks(params=sim_params)
    measurement_history = lmb.create_measurement_history(gt_target_track_history, params=sim_params)

    tracker_est_history = np.zeros(0, dtype = sim_params.dt_tracks)

    for ts in range(sim_params.sim_length):
        tracker_est = tracker.update(measurement_history[measurement_history['ts']==ts])

        tracker_est_ts = np.zeros(len(tracker_est),dtype=sim_params.dt_tracks)
        tracker_est_ts['ts'] = ts
        tracker_est_ts['label'] = tracker_est['label']
        tracker_est_ts['x'] = tracker_est['x']
        tracker_est_ts['r'] = tracker_est['r']

        tracker_est_history = np.concatenate((tracker_est_history, tracker_est_ts))

    mot_summary, mot_ts_results = lmb.evaluate_point_2D(gt_target_track_history, tracker_est_history, sim_params.max_d2)
    lmb.create_report_point_2D(gt_target_track_history, tracker_est_history, mot_summary, mot_ts_results)

if __name__ == '__main__':
    main()
