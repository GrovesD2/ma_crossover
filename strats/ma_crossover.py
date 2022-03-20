'''
Simple MA crossover strategy
'''

import random
import numpy as np

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
        df[f'{speed}_ma'] = df[price_field].ewm(span = avg_days,
                                                adjust = False).mean()
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

def get_signal_idx(df: pandasDF) -> np_arr:
    '''
    Return the indexes where a buy signal is found.
    '''
    signal = (
        (df['fast_ma'].shift(1) < df['slow_ma'].shift(1)) &
        (df['fast_ma'] > df['slow_ma'])
        )
    return np.where(signal.values)[0]

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
    
    # Check the maximum stop loss criteria
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