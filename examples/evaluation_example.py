""" 
Simple test to show functionality of evaluation.py
"""
import lmb

def main():
    
    sim_params = lmb.SimParameters() 

    gt_target_track_history = lmb.create_target_tracks(params=sim_params)

    measurement_history = lmb.create_measurement_history(gt_target_track_history, params=sim_params)
 
    lmb.evaluate(gt_target_track_history,gt_target_track_history)

if __name__ == '__main__':
    main()

