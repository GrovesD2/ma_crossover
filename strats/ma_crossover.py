'''
Simple MA crossover strategy
'''

import random
import numpy as np
import numba as nb

from typing import Tuple
from numpy import array as np_arr
from pandas import DataFrame as pandasDF

from utils import strategy as strat

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
    df_summary = strat.summarise_buy_sell(df, percs, bought, sold, hold)
    stats = strat.get_strat_stats(np.array(percs), hold)
    
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

def perturb_strat(strat: dict,
                  ga_config: dict) -> dict:
    '''
    Perturb the parameters of the strategy slightly to generate a new strategy
    '''
    
    # Lists to randomly select from
    ma_types = ['rolling', 'exp']
    price_types = ['Open', 'Low', 'High', 'Close']
    
    # Randomly select the moving average types
    strat['slow type'] = random.choice(ma_types)
    strat['fast type'] = random.choice(ma_types)
    
    # Randomly select the price field to consider
    strat['slow price'] = random.choice(price_types)
    strat['fast price'] = random.choice(price_types)
    
    # Randomly generate the number of days for the strategy
    strat['slow days'] += np.random.randint(-5, 5)
    strat['fast days'] += np.random.randint(-5, 5)
    
    # Randomly generate the profit target, stop loss, and max holding time
    strat['profit'] += np.random.uniform(-2, 2)
    strat['stop'] += np.random.uniform(-2, 2)
    strat['max hold'] += np.random.randint(-2, 2)
    
    return strat

def check_params(strat: dict,
                 ga_config: dict) -> dict:
    '''
    Check if the parameters for the strategy make sense, adjust if they dont
    '''
    
    # Check to see if the moving average days are > 2
    if strat['slow days'] < 3:
        strat['slow days'] = 3
    if strat['fast days'] < 3:
        strat['fast days'] = 3
    
    # Check if the slow days is more than the fast days
    if strat['slow days'] <= strat['fast days']:
        strat['slow days'] = strat['fast days'] + 1
        
    # Check to see if the profit is +ve and stop is -ve
    if strat['profit'] <= 0:
        strat['profit'] = 0.1
        
    if strat['stop'] >= 0:
        strat['stop'] = -0.1
        
    # Check the max-holding days is not less than a day
    if strat['max hold'] < 1:
        strat['max hold'] = 1
    
    if strat['max hold'] > ga_config['max hold']:
        strat['max hold'] = ga_config['max hold']
        
    return strat

def get_random_strat(ga_config: dict) -> dict:
    '''
    Generate a random strategy.

    Parameters
    ----------
    ga_config : dict
        The config params for the ga

    Returns
    -------
    strat : dict
        A config with the strategy params
    '''
    
    # Lists to randomly select from
    ma_types = ['rolling', 'exp']
    price_types = ['Open', 'Low', 'High', 'Close']
    
    # Initialise an empty dictionary to store the solution in
    strat = {}
    
    # Randomly select the moving average types
    strat['slow type'] = random.choice(ma_types)
    strat['fast type'] = random.choice(ma_types)
    
    # Randomly select the price field to consider
    strat['slow price'] = random.choice(price_types)
    strat['fast price'] = random.choice(price_types)
    
    # Randomly generate the number of days for the strategy
    strat['slow days'] = np.random.randint(3, 300)
    strat['fast days'] = np.random.randint(3, 300)
    
    # Randomly generate the profit target, stop stop, and max holding time
    strat['profit'] = np.random.uniform(0.1, 40)
    strat['stop'] = np.random.uniform(-40, -0.1)
    strat['max hold'] = np.random.randint(1, 300)
    
    return check_params(strat, ga_config)