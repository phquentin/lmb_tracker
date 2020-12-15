import motmetrics as mm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import matplotlib.lines as mlines
import copy


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

    NUMBER_TS = np.amax(tracks_gt['ts']) + 1
   
    for ts in range(NUMBER_TS):

        gt_labels_ts = tracks_gt[tracks_gt['ts']==ts]['label']
        est_labels_ts = tracks_est[tracks_est['ts']==ts]['label']
        
        gt_states_ts = tracks_gt[tracks_gt['ts']==ts]['x']
        est_states_ts = tracks_est[tracks_est['ts']==ts]['x']
       
        hypothesis_distance_ts = mm.distances.norm2squared_matrix(gt_states_ts, est_states_ts, max_d2=max_d2)
        frameid = acc.update(gt_labels_ts, est_labels_ts, hypothesis_distance_ts)
        mot_ts_results.append(acc.mot_events.loc[frameid])
     
    mh = mm.metrics.create()
    mot_summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
   
    return(mot_summary, mot_ts_results)


def create2D_point_report(tracks_gt, tracks_est, mot_summary, mot_ts_results):
    """
    Creates a pdf evaluation report for multi-target 2D point tracking problems. 

    Generates:

        - a plot showing the ground truth tracks and track estimates in the two 
          2D plane for the complete evaluation time with a table containing the 
          corresponding MOT-metric results.

        - a plot showing the ground truth tracks and track estimates in 3D 
          (third dimension is the time step) for the complete evaluation time with 
          a table containing the corresponding MOT-metric results.

        - a plot for each time step showing the ground truth tracks till that
          time step, the track estimates for that time step and a corresponding
          table containing the MOT-events for that time step.
    
    TODO: - incorporate clutter
          - connect markers 
    
    Parameters
    ----------
    tracks_gt: ndarray
        Array of ground truth tracks (dtype = SimParameters.dt_tracks)

    tracks_est: ndarray
        Array of estimated tracks (dtype = SimParameters.dt_tracks)

    mot_summary: pandas.DataFrame 
        Contains the MOT-metric results for the complete evaluation time

    mot_ts_results: list of pandas.DataFrame 
        Contains the MOT-events for each times step of the complete evaluation time     
    """

    NUMBER_TS = np.amax(tracks_gt['ts']) + 1
    gt_labels = np.unique(tracks_gt['label'])
    est_labels = np.unique(tracks_est['label'])

    # create colors for estimated track markers
    NUM_COLORS = len(est_labels)
    cm_subsection = linspace(0.0, 1.0, NUM_COLORS) 
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

        # MOT-metrics info for overall 2D/3D table generation
        cols = ['MOTA', 'MOTP', 'FP',  'FN', 'IDs']

        # 2D list: consecutive list represent rows, elements wihtin one list represent columns
        cells = [ [ round(mot_summary['mota']['acc'],2),  \
                    round(mot_summary['motp']['acc'],2), \
                    mot_summary['num_false_positives']['acc'], \
                    mot_summary['num_misses']['acc'], \
                    mot_summary['num_switches']['acc'] ] ]

        # Create Overall results in 2D

        # create figure and axes
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(12, 12)
        ax0 = fig.add_subplot(gs[1:9,1:-1])
        ax0.set(title ='Overall results in 2D' ,xlabel='x' ,ylabel='y')

        ax1 = fig.add_subplot(gs[9:12,1:-1])
        ax1.axis('off')
        ax1.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')

        # write ground truth into axes
        plot_gt(ax0, gt_labels, tracks_gt, NUMBER_TS, interval = False)

        # write tracker estimates into axes
        plot_track_est(ax0, est_labels, tracks_est, NUMBER_TS, colors)
  
        ax0.legend(handles=[gt_marker, track_marker], fontsize='8')                 
        pdf.savefig()
        plt.close()

        # Create Overall results in 3D

        # create figure and axes
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(12, 12)
        ax0 = fig.add_subplot(gs[1:9,1:-1], projection='3d')
        ax0.set(title ='Overall results in 3D', xlabel='x' ,ylabel='y', zlabel= 'ts')

        ax1 = fig.add_subplot(gs[9:12,1:-1])
        ax1.axis('off')
        ax1.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')
        
        # write ground truth into axes 
        plot_gt(ax0, gt_labels, tracks_gt, NUMBER_TS, d=True)

        # write tracker estimates into axes
        plot_track_est(ax0, est_labels, tracks_est, NUMBER_TS, colors, d=True)
 
        ax0.legend(handles=[gt_marker, track_marker], fontsize='8')               
        pdf.savefig()
        plt.close()
    
        # Create Time step evaluation
        
        for interval in range(NUMBER_TS):
            
            # create figure

            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(12, 12)
            ax0 = fig.add_subplot(gs[1:9,1:-1])
            title = 'Time step:{}'.format(interval)
            ax0.set(title=title ,xlabel='x' ,ylabel='y')

            # table data
            # MOT-metrics info for table generation
            cols = ['Event', 'Type', 'OId', 'HId',  'D']

            # init table data
            dim_cols = len(cols)
            dim_rows = len(mot_ts_results[interval]['Type'])
            # init 2d list: consecutive list represent rows, elements wihtin one list represent columns
            cells = [ ([0] * dim_cols) for row in range(dim_rows) ]
            # fill table data
            for i, tp in enumerate(mot_ts_results[interval]['Type']):
                cells[i][0] = i
                cells[i][1] = tp
            for i, OId in enumerate(mot_ts_results[interval]['OId']):
                cells[i][2] = OId
            for i, HId in enumerate(mot_ts_results[interval]['HId']):
                cells[i][3] = round(HId, 1)
            for i, D in enumerate(mot_ts_results[interval]['D']):
                cells[i][4] = round(D, 3)

            # create table
            ax1 = fig.add_subplot(gs[9:12,1:-1])
            print(type(ax1))
            ax1.axis('off')
            ax1.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')

            markers = []

            # write ground truth into axes  
            plot_gt(ax0, gt_labels, tracks_gt, interval, interval=True)

            # write tracker estimates into axes
            for i, label in enumerate(est_labels):
               
                track_label_data = tracks_est[tracks_est['label'] == label]
            
                x = np.squeeze(track_label_data[track_label_data['ts'] == interval ]['x'])
                    
                if(len(x) == 0):
                    continue
                marker_info = str(label) + ', r: {}'.format(round(float(track_label_data[track_label_data['ts'] == interval ]['r']), 2))
                ax0.plot(x[0], x[1], 'x', markersize=3, color = colors[i])
                markers.append(mlines.Line2D([], [], color=colors[i], marker='x', linestyle='None', markersize=5, label=marker_info))

            ax0.legend(handles=markers, fontsize='8')
            pdf.savefig()
            plt.close()


def plot_gt(ax, gt_labels, tracks_gt, NUMBER_TS, interval = False, d = False):
    """
    Plots the ground truth tracks for the given time interval of NUMBER_TS
    
    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Subplot axes for ground truth plot

    gt_labels: ndarray
        The sorted unique ground truth labels
       
    tracks_gt: ndarray 
        Array of ground truth tracks (dtype = SimParameters.dt_tracks)

    NUMBER_TS: float
        Number of time steps 

    interval: bool
        Sets whether the function is used within the interval loop to create a plot for each time step

    d: bool
        Sets whether the plot is a 3D plot     
    """

    num_iter = copy.deepcopy(NUMBER_TS)
    if(interval):
        num_iter +=1

    for label in gt_labels:
                track_label_data = tracks_gt[tracks_gt['label'] == label]

                for ts in range(num_iter):    
                    x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])

                    if(len(x) == 0):
                        continue
                    if(d):
                        if(ts != num_iter-1):
                            ax.plot(x[0], x[1], ts, 'o', markersize=3, color = 'k')
                        else:
                            # highlight last location to indicate movement direction
                            ax.plot(x[0], x[1], ts, 'o', markersize=5, color = 'k')
                    else:
                        if(ts != num_iter-1):
                            ax.plot(x[0], x[1], 'o', markersize=3, color = 'k')
                        else:
                            # highlight last location to indicate movement direction
                            ax.plot(x[0], x[1], 'o', markersize=5, color = 'k') 


def plot_track_est(ax, est_labels, tracks_est, NUMBER_TS, colors, d=False):
    """
    Plots the estimated tracks for the given time interval of NUMBER_TS
    
    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Subplot axes for estimated tracks plot

    est_labels: ndarray
        The sorted unique estimated labels
       
    tracks_est: ndarray 
        Array of estimated tracks (dtype = SimParameters.dt_tracks)

    NUMBER_TS: float
        Number of time steps 

    colors: list
        list of matplotlib colors

    d: bool
        Sets whether the plot is a 3D plot     
    """

    for i, label in enumerate(est_labels):
    
        track_label_data = tracks_est[tracks_est['label'] == label]
        
        for ts in range(NUMBER_TS):
            
            x = np.squeeze(track_label_data[track_label_data['ts'] == ts ]['x'])
            
            if(len(x) == 0):
                continue
            
            if(d):
                if(ts != NUMBER_TS-1):
                    ax.plot(x[0], x[1], ts, 'x', markersize=3, color = colors[i])
                else:
                    # highlight last location to indicate movement direction    
                    ax.plot(x[0], x[1], ts,  'x', markersize=5, color = colors[i]) 
            else:
                if(ts != NUMBER_TS-1):
                    ax.plot(x[0], x[1], '-x', markersize=3, color = colors[i])
                else:
                    # highlight last location to indicate movement direction    
                    ax.plot(x[0], x[1], '-x', markersize=5, color = colors[i]) 
