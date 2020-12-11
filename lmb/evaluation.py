import motmetrics as mm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import os


def evaluate_2D_point(tracks_gt, tracks_est, max_d2, plot = False):
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

    number_ts = np.amax(tracks_gt['ts']) + 1
   
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
    
    return(strsummary)


def plot_2D_point_results(tracks_gt, measurement_history, tracks_est, mot_summary, save_results = False):
    """
    Plots the results
    TODO: Write number_ts etc. in capital letters
    
    Parameters
    ----------
    tracks_gt: ndarray
        Array of ground truth tracks (dtype = SimParameters.dt_tracks)

    tracks_est: ndarray
        Array of estimated tracks (dtype = SimParameters.dt_tracks)

    max_d2: int 
        Maximum squared euclidian distance for which py-motmetrics creates a hypothesis between a ground truth track and estimated track   
    """

    number_ts = np.amax(tracks_gt['ts']) + 1
    gt_labels = np.unique(tracks_gt['label'])
    est_labels = np.unique(tracks_est['label'])
    print('est_labels')
    print(est_labels)
    
    # create colors for estimated tracks
    num_colors = len(est_labels)
    print('num_colors')
    print(num_colors)
    cm_subsection = linspace(0.0, 1.0, num_colors) 
    colors = [cm.Set1(x) for x in cm_subsection]
    
    # check for directory eval_results, if not there create it
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')

    # create timestep_eval
    
    for interval in range(number_ts):
    
        # write ground truth into plot  
        for label in gt_labels:

            track_label_data = tracks_gt[tracks_gt['label'] == label]

            # plot track  for current interval
            for ts in range(interval+1):
                
                x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])

                if(ts != interval):
                    plt.plot(x[0], x[1], 'o', markersize=3, color = 'k')
                else:
                # highlight last location to indicate movement direction
                    print('hrllo')
                    plt.plot(x[0], x[1], 'o', markersize=5, color = 'k') 
        
        # write tracker estimates into plot
        for i, label in enumerate(est_labels):
          
            track_label_data = tracks_est[tracks_est['label'] == label]
         
            x = np.squeeze(track_label_data[track_label_data['ts'] == interval ]['x'])
                
            if(len(x) == 0):
                continue
    
            plt.plot(x[0], x[1], 'x', markersize=3, color = colors[i])
                                                
        plt.savefig('eval_results/eval_time_step_{}.pdf'.format(interval))  
        #clear plt for next interval
        plt.clf()

    # create overall eval 2D

    # write ground truth into plot  
    for label in gt_labels:

        track_label_data = tracks_gt[tracks_gt['label'] == label]

        # plot track  for current interval
        for ts in range(number_ts):
            
            x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])

            if(ts != number_ts-1):
                plt.plot(x[0], x[1], 'o', markersize=3, color = 'k')
            else:
                # highlight last location to indicate movement direction
                print('hrllo')
                plt.plot(x[0], x[1],'o', markersize=5, color = 'k') 
    
    # write tracker estimates into plot
    for i, label in enumerate(est_labels):
        
        track_label_data = tracks_est[tracks_est['label'] == label]
        
        #plot track
        for ts in range(number_ts):
            
            x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])
            
            if(len(x) == 0):
                continue
        
            if(ts != number_ts-1):
                plt.plot(x[0], x[1], 'x', markersize=3, color = colors[i])
            else:
                # highlight last location to indicate movement direction    
                plt.plot(x[0], x[1], 'x', markersize=5, color = colors[i]) 
                                    
    plt.savefig('eval_results/2d_eval_overall.pdf')  
    #clear plt 
    plt.clf()

    #create overall 3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # write ground truth into plot  
    for label in gt_labels:

        track_label_data = tracks_gt[tracks_gt['label'] == label]

        # plot track  for current interval
        for ts in range(number_ts):
            
            x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])

            if(ts != number_ts-1):
                ax.plot(x[0], x[1], ts, 'o', markersize=3, color = 'k')
            else:
                # highlight last location to indicate movement direction
                print('hrllo')
                ax.plot(x[0], x[1], ts, 'o', markersize=5, color = 'k') 
    
    # write tracker estimates into plot
    for i, label in enumerate(est_labels):
        
        track_label_data = tracks_est[tracks_est['label'] == label]
        
        #plot track
        for ts in range(number_ts):
            
            x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])
            
            if(len(x) == 0):
                continue
        
            if(ts != number_ts-1):
                ax.plot(x[0], x[1], ts, 'x', markersize=3, color = colors[i])
            else:
                # highlight last location to indicate movement direction    
                ax.plot(x[0], x[1], ts,  'x', markersize=5, color = colors[i]) 
                                    
    plt.savefig('eval_results/3d_eval_overall.pdf')  
    #clear plt 
    plt.clf()

    print(mot_summary)
