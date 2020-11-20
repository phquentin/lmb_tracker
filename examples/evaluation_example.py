""" 
Simple example to show functionality of evaluation.py
"""
import lmb

def main():
    
    sim_params = lmb.SimParameters() 

    gt_target_track_history = lmb.create_target_tracks(params=sim_params)
 
    lmb.evaluate2Dpoint(gt_target_track_history, gt_target_track_history, sim_params.max_d2)

if __name__ == '__main__':
    main()

