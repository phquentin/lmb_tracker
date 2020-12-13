import motmetrics as mm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import matplotlib.lines as mlines


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
    mot_ts_results = []

    number_ts = np.amax(tracks_gt['ts']) + 1
   
    for ts in range(number_ts):

        gt_labels_ts = tracks_gt[tracks_gt['ts']==ts]['label']
        est_labels_ts = tracks_est[tracks_est['ts']==ts]['label']
        
        gt_states_ts = tracks_gt[tracks_gt['ts']==ts]['x']
        est_states_ts = tracks_est[tracks_est['ts']==ts]['x']
       
        hypothesis_distance_ts = mm.distances.norm2squared_matrix(gt_states_ts, est_states_ts, max_d2=max_d2)
        frameid = acc.update(gt_labels_ts, est_labels_ts, hypothesis_distance_ts)
        mot_ts_results.append(acc.mot_events.loc[frameid])
        print(acc.mot_events.loc[frameid])

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    #print(summary)
    #print(summary['mota'])
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    
    return(strsummary, mot_ts_results)


def create2D_point_report(tracks_gt, measurement_history, tracks_est, mot_summary, mot_ts_results):
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

    # create colors for estimated tracks
    num_colors = len(est_labels)
    cm_subsection = linspace(0.0, 1.0, num_colors) 
    colors = [cm.Set1(x) for x in cm_subsection]
    
    # check for directory eval_results, if not there create it
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')

    # create report name
    current_date = datetime.now()
    date_time = current_date.strftime("%Y_%m_%d_%H:%M:%S")
    report_name = 'eval_results/eval_report_{}.pdf'.format(date_time)
   
   # create pdf report
    with PdfPages(report_name) as pdf:

        gt_marker = mlines.Line2D([], [], color='k',marker='o',linestyle='None', markersize=5, label='ground truth')
        track_marker = mlines.Line2D([], [], color='k',marker='x',linestyle='None', markersize=5, label='track estimate')

        # Create overall eval 2D

        # create figure
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.25)
        ax = fig.add_subplot(111)
        ax.set(title ='Overall results in 2D' ,xlabel='x' ,ylabel='y')
        ax.text(0.5, 0.1, mot_summary, transform=fig.transFigure, horizontalalignment='center', verticalalignment='center', size=8)

        # write ground truth into plot
        for label in gt_labels:

            track_label_data = tracks_gt[tracks_gt['label'] == label]

            # plot track  for current interval
            for ts in range(number_ts):
                
                x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])

                if(ts != number_ts-1):
                    ax.plot(x[0], x[1], 'o', markersize=3, color ='k')
                else:
                    # highlight last location to indicate movement direction
                    ax.plot(x[0], x[1],'o', markersize=5, color ='k') 
        
        # write tracker estimates into plot
        for i, label in enumerate(est_labels):
            
            track_label_data = tracks_est[tracks_est['label'] == label]
            
            #plot track
            for ts in range(number_ts):
                
                x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])
                
                if(len(x) == 0):
                    continue
            
                if(ts != number_ts-1):
                    ax.plot(x[0], x[1], 'x', markersize=3, color = colors[i])
                else:
                    # highlight last location to indicate movement direction    
                    ax.plot(x[0], x[1], 'x', markersize=5, color = colors[i]) 

        plt.legend(handles=[gt_marker, track_marker], fontsize='8')                 
        pdf.savefig()
        plt.close()

        # Create overall 3D

        fig = plt.figure()
        fig.subplots_adjust(bottom=0.25)
        ax = fig.add_subplot(111, projection='3d')
        ax.set(title ='Overall results in 3D', xlabel='x' ,ylabel='y', zlabel= 'ts')
        ax.text2D(0.5, 0.1, mot_summary, transform=fig.transFigure, horizontalalignment='center', verticalalignment='center', size=8)
        
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

        plt.legend(handles=[gt_marker, track_marker], fontsize='8')               
        pdf.savefig()
        plt.close()
    
        # Create timestep_eval
        
        for interval in range(number_ts):
            
            # create figure
            fig = plt.figure()
            fig.subplots_adjust(bottom=0.35)
            ax = fig.add_subplot(111)
            title = 'Time step:{}'.format(interval)
            ax.set(title=title, xlabel='x' ,ylabel='y')
            text = str(mot_ts_results[interval])
            ax.text(0.5, 0.15, text, transform=fig.transFigure, horizontalalignment='center', verticalalignment='center', size=8)
            markers = []
            # write ground truth into plot  
            for label in gt_labels:
                
                track_label_data = tracks_gt[tracks_gt['label'] == label]

                # plot track  for current interval
                for ts in range(interval+1):
                    
                    x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])

                    if(ts != interval):
                        ax.plot(x[0], x[1], 'o', markersize=3, color = 'k')
                    else:
                        # highlight last location to indicate movement direction
                        ax.plot(x[0], x[1], 'o', markersize=5, color = 'k') 
            
            # write tracker estimates into plot
            for i, label in enumerate(est_labels):
               
                track_label_data = tracks_est[tracks_est['label'] == label]
            
                x = np.squeeze(track_label_data[track_label_data['ts'] == interval ]['x'])
                    
                if(len(x) == 0):
                    continue
                marker_info = str(label) + ', r: {}'.format(round(float(track_label_data[track_label_data['ts'] == interval ]['r']), 2))
                ax.plot(x[0], x[1], 'x', markersize=3, color = colors[i])
                markers.append(mlines.Line2D([], [], color=colors[i], marker='x', linestyle='None', markersize=5, label=marker_info))

            plt.legend(handles=markers, fontsize='8')
            pdf.savefig()
            plt.close()
