import os
from shutil import copyfile
import numpy as np
from numpy import linspace
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

import motmetrics as mm


def evaluate_point_2D(tracks_gt, tracks_est, max_d2, plot = False):
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

    Returns
    -------
    out: (pandas.DataFrame, list of pandas.DataFrame)
        (Contains the MOT-metric results for the complete evaluation time, 
         Contains the MOT-events for each times step of the complete evaluation time)
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


def prepare_results_dir(unique_dir_name):
    """
    Creates the results folder and saves the filter parameter file into it.

    Parameters
    ----------
    unique_dir_name: str
        Unique name of the folder storing the results

    Returns
    -------
    out: str
        Relative path of the created folder
    """
    results_folder = 'eval_results'
    
    # check for directory eval_results, if not there create it
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    dir_name = os.path.join(results_folder, unique_dir_name)
    # check for directory of results, if not there create it
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    
    # Copy filter parameter file into results directory
    param_dir = 'lmb'
    param_file = 'parameters.py'
    try:
        copyfile(os.path.join(param_dir, param_file), os.path.join(dir_name, param_file))
    except(FileNotFoundError):
        print("Parameter file {} not found. Unable to copy.".format(os.path.join(param_dir, param_file)))
    
    return dir_name


def create_report_point_2D(tracks_gt, tracks_est, mot_summary, mot_ts_results):
    """
    Creates a pdf evaluation report for multi-target 2D point tracking problems
    and saves it in /examples/eval_results

    Generates:

        - a plot showing the ground truth tracks and track estimates in the two 
          2D plane for the complete evaluation time with a table containing the 
          corresponding MOT-metric results.

        - a plot for each time step showing the ground truth tracks till that
          time step, the track estimates for that time step and a corresponding
          table containing the MOT-events for that time step.
    
    TODO: - incorporate clutter
    
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
    print("\n Generating evaluation report...")

    NUMBER_TS = np.amax(tracks_gt['ts']) + 1
    gt_labels = np.unique(tracks_gt['label'])
    est_labels = np.unique(tracks_est['label'])

    # create colors for estimated track markers
    NUM_COLORS = len(est_labels)
    cm_subsection = linspace(0.0, 1.0, NUM_COLORS) 
    colors = [cm.Set1(x) for x in cm_subsection]
    
    # create report name
    current_date = datetime.now()
    date_time = current_date.strftime("%Y_%m_%d_%H-%M-%S")
    report_dir = prepare_results_dir(date_time)
    report_name = 'eval_report_{}.pdf'.format(date_time)
   
   # create pdf report
    with PdfPages(os.path.join(report_dir, report_name)) as pdf:

        gt_marker = mlines.Line2D([], [], color='k',marker='o',linestyle='dashed', linewidth=1, markersize=4, label='ground truth')
        track_marker = mlines.Line2D([], [], color='k',marker='x',linestyle='dashed', linewidth=1, markersize=4, label='track estimate')

        # MOT-metrics info for overall 2D table generation
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
        plot_tracks(ax0, gt_labels, tracks_gt, NUMBER_TS, gt_marker)

        # write tracker estimates into axes
        plot_tracks(ax0, est_labels, tracks_est, NUMBER_TS, track_marker, colors)
  
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
            ax1.axis('off')
            ax1.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')

            markers = []

            # write ground truth into axes  
            plot_tracks(ax0, gt_labels, tracks_gt, interval, gt_marker)

            # write tracker estimates into axes
            for i, label in enumerate(est_labels):
                track_label_data = tracks_est[tracks_est['label'] == label]
            
                x = np.squeeze(track_label_data[track_label_data['ts'] == interval ]['x'])
                    
                if(len(x) == 0):
                    continue
                marker_info = str(label) + ', r: {}'.format(round(float(track_label_data[track_label_data['ts'] == interval ]['r']), 2))
                ax0.plot(x[0], x[1], marker=track_marker.get_marker(), markersize=4, color = colors[i])
                markers.append(mlines.Line2D([], [], color=colors[i], marker=track_marker.get_marker(),\
                    linestyle='None', markersize=5, label=marker_info))

            ax0.legend(handles=markers, fontsize='8')
            pdf.savefig()
            plt.close()

    print("\n Report saved as {}".format(report_name))


def plot_tracks(ax, labels, tracks, ts, format, colors=None):
    """
    Plots the tracks from start until the given timestep

    If no colors are given, the tracks are plotted in black.
    
    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Subplot axes for estimated tracks plot

    labels: ndarray
        The sorted unique estimated labels
       
    tracks: ndarray 
        Array of tracks (dtype = SimParameters.dt_tracks)

    ts: float
        Timestep

    format: matplotlib.lines.line2D
        Line format object to define linestyle, linewidth, and marker
    
    colors: list, optional
        list of matplotlib colors, same length as labels  
    """
    for i, label in enumerate(labels):
        track = tracks[tracks['label'] == label]
        x = track[track['ts'] <= ts]['x']

        if(len(x) > 0):
            color = colors[i] if colors is not None else 'k'
            ax.plot(x[:,0], x[:,1], linestyle=format.get_linestyle(), \
                linewidth=format.get_linewidth(), marker=format.get_marker(), \
                    markersize=4, color=color)
            # plot bigger marker at last timestep
            ax.plot(x[-1,0], x[-1,1], linestyle=format.get_linestyle(), \
                linewidth=format.get_linewidth(), marker=format.get_marker(), \
                    markersize=6, color=color)
