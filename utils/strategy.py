'''
Any strategy dependent functionality
'''

import numpy as np
import numba as nb

from typing import Tuple
from numpy import array as np_arr
from pandas import DataFrame as pandasDF

def add_ma_col(df: pandasDF,
               speed: str,
               mean_type: str,
               price_field: str,
               avg_days: int) -> pandasDF:
    '''
    Add the moving average column to the dataframe

    Parameters
    ----------
    df : pandasDF
        Price data
    speed : str 
        The string representation for the speed of the ma
    mean_type: str
        To determine if the moving average is rolling, or exponential
    price_field: str
        Which price column to consider
    avg_days: int
        How many days to include on the rolling average

    Returns
    -------
    df: pandasDF
        The input df with the moving average column calculated
    '''
    
    if mean_type == 'exp':
        df[f'{speed}_ma'] = df[price_field].ewm(avg_days).mean()
    else:
        df[f'{speed}_ma'] = df[price_field].rolling(avg_days).mean()
    
    return df

def add_strat_cols(df: pandasDF,
                   config: dict) -> pandasDF:
    '''
    Add the moving average columns to the dataframe, based on the config dict.
    Only configured for two moving averages for now

    Parameters
    ----------
    df : pandasDF
        Dataframe of daily price data.
    config : dict
        Moving average parameters.

    Returns
    -------
    df : pandasDF
        Dataframe with the moving average cols included.
    '''
    df = add_ma_col(df,
                    'slow',
                    config['slow type'], 
                    config['slow price'],
                    config['slow days'])
    
    df = add_ma_col(df,
                    'fast',
                    config['fast type'], 
                    config['fast price'],
                    config['fast days'])
    
    return df.dropna().reset_index().drop(columns = 'index')

def run_strategy(df: pandasDF,
                 config: dict) -> Tuple[pandasDF, dict]:
    '''
    Run the strategy on the current ticker and with the specified config

    Parameters
    ----------
    df : pandasDF
        The price data
    config : dict
        The configuration settings for the strategy

    Returns
    -------
    df_summary : pandasDF
        The dates of each trade, along with the profit and trade-days held
    stats: dict
        The key metrics to judge the strategy by 

    '''
    # Perform the buying and selling
    percs, bought, sold = make_trades(df['Open'].values.astype(np.float64),
                                      df['Low'].values.astype(np.float64),
                                      df['High'].values.astype(np.float64),
                                      df['slow_ma'].values.astype(np.float64),
                                      df['fast_ma'].values.astype(np.float64),
                                      config['profit'],
                                      config['stop'],
                                      config['max hold'])
    
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
                slow_ma: np_arr,
                fast_ma: np_arr,
                profit: float,
                stop: float,
                max_hold: float) -> Tuple[list, list, list]:
    '''
    Run the ma crossover strategy. The rules are simple, if the fast ma crosses
    the slow from underneath then the stock is bought on the next open. It is
    sold either when the profit target or stop loss is hit.

    Parameters
    ----------
    Open, Low, High : np_arr
        The open/low/high stock-prices
    slow_ma, fast_ma : np_arr
        The moving averages
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
    
    # This is a flag to show if we are holding a stock or not, so the correct
    # if statements are entered in the buy/sell algorithm
    holding = False
    
    for n in range(2, len(Open)):
        
        if not holding:
            
            # This sets the crossover, the fast must cross the slow from
            # underneath it
            if fast_ma[n-2] < slow_ma[n-2] and fast_ma[n-1] > slow_ma[n-1]:
                
                # Sometimes zero-price data exists. To avoid any divisions by
                # zero when calculating percentages, these are not included.
                if Open[n] > 0:
                    price = Open[n]
                    bought.append(n)
                    
                    # Check if the profit/stop are hit on the same day
                    # NOTE: Sometimes both can happen, but this is such a rare
                    # event that it does not harm the input data to the nn
                    if 100*(High[n]/price - 1) >= profit:
                        percs.append(profit)
                        sold.append(n)
                
                    elif 100*(Low[n]/price - 1) <= stop:
                        percs.append(stop)
                        sold.append(n)
                        
                    else:
                        holding = True
                    
                    continue
            
        if holding:
            # If we are holding the stock, check to see if the profit target
            # or stop loss is hit (done in percentages)
            
            if 100*(High[n]/price - 1) >= profit:
                percs.append(profit)
                sold.append(n)
                holding = False
            elif 100*(Low[n]/price - 1) <= stop:
                percs.append(stop)
                sold.append(n)
                holding = False
            elif n - bought[-1] > max_hold:
                percs.append(100*(Open[n]/price - 1))
                sold.append(n)
                holding = False
            
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