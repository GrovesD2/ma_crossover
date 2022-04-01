'''
Any random utilities for the NN training/testing
'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# For type hinting
from numpy import array as np_arr

# Project imports
from utils import io

def add_gen_config(nn_config: dict) -> dict:
    '''
    Add the params from the config settings used to generate the NN train data

    Parameters
    ----------
    nn_config : dict
        The config used to train the NN.

    Returns
    -------
    nn_config : dict
        The config with the generation settings
    '''
    
    # Load the config used to generate the generate the NN data
    gen_config = io.load_dict('nn/data/' + nn_config['strat name'])
    
    # Add the configs to the NN config
    nn_config['lower date filter'] = gen_config['lower date filter']
    nn_config['include fundamentals'] = gen_config['include fundamentals']
    nn_config['time lags'] = gen_config['time lags']
    
    return nn_config

def get_confusion_matrix(true_labels: np_arr,
                         pred_labels: np_arr,
                         nn_config: dict,
                         title: str):
    '''
    Get the confusion matrix plot and save it in the NN models folder

    Parameters
    ----------
    true_labels : np_arr
        The true labels
    pred_labels : np_arr
        The labels predicted by the NN.
    nn_config : dict
        The configuration settings for the NN
    title : str
        The title for the plot.

    Returns
    -------
    None
    '''    
    cm = confusion_matrix(y_true = true_labels,
                          y_pred = pred_labels)
    #cm_scaled = cm/cm.astype(np.float).sum(axis = 0)
    cm_scaled = cm
    
    disp = ConfusionMatrixDisplay(confusion_matrix = cm_scaled)
    disp.plot()
    
    plt.title(title)
    plt.savefig('nn/models/' + nn_config['model save name'] +
               ' ' + title.lower() + '.png')

    return