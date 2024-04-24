import os
import pickle
import logging
import bz2
import sys

import matplotlib as mpl
import cvxopt
import numpy as np
import GPy



import matplotlib.pyplot as plt
from matplotlib import font_manager


def get_versions():

    versions = {
        'cvxopt': cvxopt.__version__,
        'GPy': GPy.__version__,
        'matplotlib': mpl.__version__,
        'numpy': np.__version__,
        'Python': sys.version,
        }
    return versions

def gen_log(log, file_path):
    '''
    Create log file of the experiment you conduct.
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    logger.debug(log)


def save_data(filename, data, logging=True):
    '''
    This method saves data at filename.
    If logging is True, this method also create a log file at the place, log_path.
    The log contains all values of parameters of the data except for that of 'repeated_history' 
    and you choose.

    inputs:
    -------
    filename: str
    data: dictionary
    logging: bool 
    
    Outputs:
    --------
    None

    '''
    except_list = ['repeated_history', 'upper_x0', 'seeds']
    dirname = os.path.dirname(filename)
    if len(dirname) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if logging:
        log_path = 'log/save_log.txt'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with bz2.open(filename, 'wb') as f:
        pickle.dump(data, f)
    if logging:
        log = ''
        for key in data:
            if key not in except_list:
                log += f'{key} : {data[key]}, ' 
        gen_log(log, log_path)


def load(file_name):
    """
    Load data a file generated by an experiment.
    """
    exceptions = ['repeated_history','upper_x0', 'elapsed_time']
    with bz2.open(file_name, 'rb') as f:
        result = pickle.load(f)
    for key, value in result.items():
        if key not in exceptions:
            print(f"{key}:{value}")
    return result


def ax_plot_revenue_episodes(ax, repeated_histories, optimal_value, labels, exceptions, mark_points, 
                             colors, markers, linestyles,
                             title=None, xlabel=None, ylabel=None, y_ticks=None, episode_slice=None, cum=True,
                             markersize=18, linewidth=5, xscale='log' ):
    """
    Plot the regrets of experiment results as in the paper.
    """
    
    # font_files = font_manager.findSystemFonts(fontpaths=['fonts/'])
    # for font_file in font_files:
    #     font_manager.fontManager.addfont(font_file)
    # mpl.rc('font', family='IPAexGothic')
    
    if len(repeated_histories) != len(labels):
        raise RuntimeError
    
    episode_slice = episode_slice if episode_slice is not None else slice(None)
    
    #Definitions of function
    def make_revenue_trials(repeated_history):
        #Cumulative sum of the revenue of each trial.
        n_trials = len(repeated_history)
        return np.vstack(
            [np.cumsum([repeated_history[cnt_trial][cnt_episode][-1]['revenue']
                     for cnt_episode in
                     range(len(repeated_history[cnt_trial]))])
             for cnt_trial in range(n_trials)])
    # Collect total revenue up to each episode for each trials.
    def mean_revenue_over_episodes(repeated_history):
        # total revenue up to each episode
        revenue_array = np.vstack([[epi_history[-1]['revenue'] for epi_history in trial] 
                         for trial in repeated_history])
        # return the average of total revenue up to each episode in trial.
        return np.mean(revenue_array, axis=0)[0]
    
    # Set ax
    ax.grid(linewidth=2)
    # Set the horizontal scale in logarithm.
    ax.set_xscale(xscale)
    # Plot the regret of each history and store it in line_list.
    line_list = []
    for cnt_history, repeated_history in enumerate(repeated_histories):
        # get an array of revenues 
        revenue_trials = make_revenue_trials(repeated_history)
        if labels[cnt_history] not in exceptions:
            n_episodes = revenue_trials.shape[1]
        # Restrict the array of revenue within the input slice 
        revenue_episode = revenue_trials[:, episode_slice]
        # Calc the avg revenue in trials.
        revenue_episode_mean = np.mean(revenue_episode, axis=0)
        # Calc the standard deviation in trials.
        revenue_episode_std = np.std(revenue_episode, axis=0)
        x = np.arange(1, n_episodes + 1)[episode_slice]
        # Exceptions for the oracle algorithms
        if labels[cnt_history] in exceptions:
            mean = mean_revenue_over_episodes(repeated_history)
            revenue_episode_mean = x * mean
        #Optimal cumulative revenue
        optimal_episodes = x * optimal_value 
        
        #Plot the regret.   
        ax_line = ax.plot(x, (1- revenue_episode_mean / optimal_episodes) * 100.0,
        label=labels[cnt_history], markersize=markersize, markevery=mark_points,color=colors[cnt_history],
        marker=markers[cnt_history], linestyle=linestyles[cnt_history], linewidth=linewidth )
        #Set values of ylabel in percent.
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
        
        # Plot the standard deviation areas of the regret.
        if labels[cnt_history] not in exceptions:
            ax.fill_between(x, (1- (revenue_episode_mean + revenue_episode_std )/ optimal_episodes) * 100.0,
                        (1- (revenue_episode_mean - revenue_episode_std )/ optimal_episodes) * 100.0, 
                        color=colors[cnt_history],label='_nolegend_', alpha=0.2)              
    

        line_list.append(ax_line[0])
    
    if xlabel is None:
        ax.set_xlabel(r'Episodes ($s$)', fontsize=25)
    else:
        ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel('Regret', fontsize=27)
    else:
        ax.set_ylabel(ylabel)
    
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    
    ax.set_title(title, fontsize=22)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height*1.1])

    return ax,line_list