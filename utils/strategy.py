'''
Any common strategy functionality
'''

import numpy as np

import strats as strat_lib

from typing import Tuple
from numpy import array as np_arr
from pandas import DataFrame as pandasDF

def add_strat_cols(df: pandasDF,
                   config: dict,
                   strat: str) -> pandasDF:
    '''
    Add the strategy columns to the dataframe

    Parameters
    ----------
    df : pandasDF
        Dataframe of daily price data.
    config : dict
        Strategy parameters.
    strat : str
        The name of the strategy

    Returns
    -------
    df : pandasDF
        Dataframe with the strat cols included.
    '''
    if strat == 'simple ma crossover':
        return strat_lib.ma_crossover.add_strat_cols(df, config)
    if strat == 'simple bollinger band':
        return strat_lib.boll_band.add_strat_cols(df, config)
    
def run_strategy(df: pandasDF,
                 config: dict,
                 strat: str) -> Tuple[pandasDF, dict]:
    '''
    Run the strategy on the current ticker and with the specified config

    Parameters
    ----------
    df : pandasDF
        The price data
    config : dict
        The configuration settings for the strategy
    strat : str
        The name of the strategy
        
    Returns
    -------
    df_summary : pandasDF
        The dates of each trade, along with the profit and trade-days held
    stats: dict
        The key metrics to judge the strategy by
    '''
    if strat == 'simple ma crossover':
        return strat_lib.ma_crossover.run_strategy(df, config)
    if strat == 'simple bollinger band':
        return strat_lib.boll_band.run_strategy(df, config)
    
def summarise_buy_sell(df: pandasDF,
                       percs: list,
                       bought: list,
                       sold: list,
                       hold: np_arr) -> pandasDF:
    '''
    Parameters
    ----------
    df : pandasDF
        Price data
    percs : list
        Profit/losses from the trades made
    bought : list
        Indexes of the dates where the stock was bought
    sold : list
        Indexes of the dates where the stock was sold
    hold: np_arr
        The holding time for the stock

    Returns
    -------
    df : pandasDF
        A dataframe summarising each buy and sell
    '''
    dates = df['Date'].values
    return pandasDF({'Bought': dates[bought],
                     'Sold': dates[sold],
                     'Profit/Loss': percs,
                     'Days held': hold})

def get_strat_stats(percs: np_arr,
                    hold: np_arr) -> dict:
    '''
    From the buying and selling, produce key performance statistics

    Parameters
    ----------
    percs : np_arr
        The profit/loss on each trade
    hold : np_arr
        The time held for each trade

    Returns
    -------
    stats: dict
        The statistics from this simulation
    '''
    
    if percs.shape[0] > 0:
        return{'win rate': 100*percs[percs>0].shape[0]/percs.shape[0],
               'avg profit': np.mean(percs),
               'median profit': np.median(percs),
               'mean hold': np.mean(hold),
               'median hold': np.median(hold),
               'min hold': np.min(hold),
               'max hold': np.max(hold),
               'number of trades': percs.shape[0]
               } 
    else:
        return{'win rate': 0,
               'avg profit': 0,
               'median profit': 0,
               'mean hold': 0,
               'median hold': 0,
               'min hold': 0,
               'max hold': 0,
               'number of trades': 0
               } 