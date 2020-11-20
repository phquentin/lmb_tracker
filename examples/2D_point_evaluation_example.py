""" 
Simple 2D point evaluation example to test and evaluate tracker for basic functionality
No full functionality since the tracker is not fully implemented. Uncomment lmb.evaluate_point_2D, when tracker is 
implemented.
"""

import lmb

def main():
    
    sim_params = lmb.SimParameters() 
    tracker_params = lmb.TrackerParameters()

    tracker = lmb.LMB(params=tracker_params)

    gt_target_track_history = lmb.create_target_tracks(params=sim_params)
    measurement_history = lmb.create_measurement_history(gt_target_track_history, params=sim_params)

    tracker_estimates_history = []
    for ts in range(sim_params.sim_length):
        tracks_estimates_ts = tracker.update(measurement_history[measurement_history['ts']==ts])
        tracker_estimates_history.append(tracks_estimates_ts)

    #lmb.evaluate_point_2D(gt_target_track_history, tracker_estimates_history, sim_params.max_d2)

if __name__ == '__main__':
    main()
