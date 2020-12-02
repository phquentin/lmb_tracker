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

    tracker_estimates_history = np.zeros(0, dtype = sim_params.dt_tracks)

    for ts in range(sim_params.sim_length):
        tracker_estimates = tracker.update(measurement_history[measurement_history['ts']==ts])
        tracker_estimates_ts = np.zeros(len(tracker_estimates),dtype=sim_params.dt_tracks)
        tracker_estimates_ts['ts'] = ts
        tracker_estimates_ts['label'] = tracker_estimates['label']
        tracker_estimates_ts['x'] = tracker_estimates['x']

        tracker_estimates_history = np.concatenate((tracker_estimates_history, tracker_estimates_ts))

    lmb.evaluate_point_2D(gt_target_track_history, tracker_estimates_history, sim_params.max_d2)

if __name__ == '__main__':
    main()
