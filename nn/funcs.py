'''
Functions for the neural network
'''

import numpy as np
import pandas as pd

from utils import tickers
from utils import strategy

from pandas import DataFrame as pandasDF


def main(nn_config: dict,
         strat_config: dict):
    '''
    Main function to run the nn

    Parameters
    ----------
    nn_config : dict
        Configuration parameters for the nn
    strat_config : dict
        Configuration parameters for the solver

    Returns
    -------
    None
    '''
    
    return

def get_nn_data(n_tickers: int) -> pandasDF:
    '''
    Given a number of tickers, produce the input data to the nn

    Parameters
    ----------
    n_tickers : int
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.
    '''
    
    ticker_list = tickers.get_random_tickers(n_tickers)
    
    # For each ticker in the ticker list, run the buy/sell and pre-process
    # the data for input into the nn
    


