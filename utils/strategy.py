'''
Any common strategy functionality
'''

import numpy as np
import numba as nb

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
    
def get_buy_signals(df: pandasDF,
                    strat: str) -> np_arr:
    '''
    Return the indexes where a buy signal is found.
    '''
    if strat == 'simple ma crossover':
        return strat_lib.ma_crossover.get_signal_idx(df)
    if strat == 'simple bollinger band':
        return strat_lib.boll_band.get_signal_idx(df)
    
def run_strategy(df: pandasDF,
                 config: dict,
                 strat_name: str) -> Tuple[pandasDF, dict]:
    '''
    Run the strategy on the current ticker and with the specified config

    Parameters
    ----------
    df : pandasDF
        The price data
    config : dict
        The configuration settings for the strategy
    strat_name : str
        The name of the strategy we are considering

    Returns
    -------
    df_summary : pandasDF
        The dates of each trade, along with the profit and trade-days held
    stats: dict
        The key metrics to judge the strategy by 
    '''
    
    # Get the indexes of the buy signals
    signal_idx = get_buy_signals(df, strat_name)

    # Perform the buying and selling
    percs, bought, sold = make_trades(df['Open'].values.astype(np.float64),
                                      df['Low'].values.astype(np.float64),
                                      df['High'].values.astype(np.float64),
                                      signal_idx.astype(np.int64),
                                      config['profit'],
                                      config['stop'],
                                      config['max hold'],
                                      )
    
    # Calculate the hold time metric
    hold = np.array(sold) - np.array(bought)
    
    # Get the summaries and return the results
    df_summary = summarise_buy_sell(df, percs, bought, sold, hold)
    stats = get_strat_stats(np.array(percs), hold)
    
    return df_summary, stats
    
@nb.jit(nopython = True)
def make_trades(Open: np_arr,
                Low: np_arr,
                High: np_arr,
                signal_idx: np_arr,
                profit: float,
                stop: float,
                max_hold: float) -> Tuple[list, list, list]:
    '''
    Run the boll band. The rules are simple, buy on the next open if the stock
    closes below the lower band, and wait until the profit target/stop loss/
    max days are hit

    Parameters
    ----------
    Open, Low, High: np_arr
        The open/low/high stock-prices
    signal_idx: np_arr
        The indexes where a buy signal was noticed on the close
    profit, stop : float
        The profit target and stop loss (in percentages)
    max_hold : float
        The maximum number of days to hold the stock for

    Returns
    -------
    percs : list
        Percentage profit/loss from each trade
    bought, sold : list
        The indexes in the dataframe at which the 
        DESCRIPTION.
        
    Notes
    -----
    There is likely an efficient way to do this in pandas, I coded it this way
    to get some experience using numba (besides, the numba compiled code is 
    pretty quick).
    '''
    
    # Initialise lists to store the data
    percs = []
    bought = []
    sold = []
    
    for signal in signal_idx:
        
        # Avoiding penny-stock behaviours
        if Open[signal] > 1 and signal + max_hold + 1 < Open.shape[0]:
            price = Open[signal + 1]
            bought.append(signal + 1)
            
            for day in range(1, max_hold + 2):
                idx = signal + day
                
                if 100*(High[idx]/price - 1) >= profit:
                    percs.append(profit)
                    sold.append(idx)
                    break
                    
                elif 100*(Low[idx]/price - 1) <= stop:
                    percs.append(stop)
                    sold.append(idx)
                    break
                
                elif day == max_hold + 1:
                    percs.append(100*(Open[idx]/price - 1))
                    sold.append(idx)
                
    # After the buying and selling has terminated, eliminate any final buy
    # signals. We are only interested in completed trades
    if len(bought) > len(percs):
        del bought[-1]
        
    return percs, bought, sold
    
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