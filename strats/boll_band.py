'''
Simple bollinger band strategy
'''

import random
import numpy as np
import numba as nb

from typing import Tuple
from numpy import array as np_arr
from pandas import DataFrame as pandasDF

from utils import strategy as strat

def add_boll_col(df: pandasDF,
                 col_name: str,
                 mean_price_field: str,
                 std_price_field: str,
                 mean_type: str,
                 std_type: str,
                 mean_days: int,
                 std_days: int,
                 boll_fact: float) -> pandasDF:
    '''
    Parameters
    ----------
    df : pandasDF
        Price data.
    col_name : str
        Name of the column to add to the df
    mean_price_field : str
        Which price column to consider for the mean
    std_price_field : str
        Which price column to consider for the std dev
    mean_type : str
        To determine if the moving average is rolling, or exponential.
    std_type : str
        To determine if the std dev is rolling, or exponential.
    mean_days : int
        Number of days to calculate the mean
    std_days : int
        Number of days to calculate the std dev
    boll_fact : float
        Factor to multiply the std dev with

    Returns
    -------
    df : pandasDF
        The price data with a bollinger band column added
    '''
    if mean_type == 'exp':
        mean = df[mean_price_field].ewm(mean_days).mean()
    else:
        mean = df[mean_price_field].rolling(mean_days).mean()

    if std_type == 'exp':
        std = df[std_price_field].ewm(std_days).std()
    else:
        std = df[std_price_field].rolling(std_days).std()
        
    df[col_name] = mean + boll_fact*std
    
    return df

def add_strat_cols(df: pandasDF,
                   config: dict) -> pandasDF:
    '''
    Add the columns necessary for this strategy.

    Parameters
    ----------
    df : pandasDF
        Dataframe of daily price data.
    config : dict
        Configuration parameters for the strategy.

    Returns
    -------
    df : pandasDF
        Dataframe with the strategy columns included.
    '''

    df = add_boll_col(df,
                      'boll_lower',
                      config['mean price'],
                      config['std price'],
                      config['mean type'],
                      config['std type'],
                      config['mean days'],
                      config['std days'],
                      config['factor'],
                      )
    
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
                                      df['Close'].values.astype(np.float64),
                                      df['boll_lower'].values.astype(np.float64),
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
                Close: np_arr,
                boll_lower: np_arr,
                profit: float,
                stop: float,
                max_hold: float) -> Tuple[list, list, list]:
    '''
    Run the boll band. The rules are simple, buy on the next open if the stock
    closes below the lower band, and wait until the profit target/stop loss/
    max days are hit

    Parameters
    ----------
    Open, Low, High, Close: np_arr
        The open/low/high stock-prices
    boll_lower : np_arr
        The lower bollinger band (for the buy signal)
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
            
            # The buy signal is when the price closes below the bollinger band
            if Close[n-1] <= boll_lower[n-1]:
                
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
    strat['mean type'] = random.choice(ma_types)
    strat['std type'] = random.choice(ma_types)
    
    # Randomly select the price field to consider
    strat['mean price'] = random.choice(price_types)
    strat['std price'] = random.choice(price_types)
    
    # Randomly perturb the number of days for the strategy
    strat['mean days'] += np.random.randint(-5, 5)
    strat['std days'] += np.random.randint(-5, 5)
    
    # Perturb the standard deviation factor
    strat['factor'] += np.random.uniform(-0.5, 0.5)
     
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
    if strat['mean days'] < 3:
        strat['mean days'] = 3
    if strat['std days'] < 3:
        strat['std days'] = 3
        
    # Check the bollinger band factor isn't unreasonable
    if strat['factor'] >= 0:
        strat['factor'] = -0.1
        
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
        
    # Check the maximum stop-loss criteria
    if strat['stop'] <= ga_config['max stop']:
        strat['stop'] = ga_config['max stop']
        
    # Check the minimum profit target criteria
    if strat['profit'] <= ga_config['min profit']:
        strat['profit'] = ga_config['min profit']
        
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
    strat['mean type'] = random.choice(ma_types)
    strat['std type'] = random.choice(ma_types)
    
    # Randomly select the price field to consider
    strat['mean price'] = random.choice(price_types)
    strat['std price'] = random.choice(price_types)
    
    # Randomly generate the number of days for the strategy
    strat['mean days'] = np.random.randint(3, 300)
    strat['std days'] = np.random.randint(3, 300)
    
    # Randomly generate a bollinger band factor
    strat['factor'] = np.random.uniform(-3, -0.1)
    
    # Randomly generate the profit target, stop stop, and max holding time
    strat['profit'] = np.random.uniform(0.1, 40)
    strat['stop'] = np.random.uniform(-40, -0.1)
    strat['max hold'] = np.random.randint(1, 300)
    
    return check_params(strat, ga_config)