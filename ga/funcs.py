'''
Functionality for running the genetic algorithm
'''

import os
import pickle
import random
import numpy as np
import pandas as pd

# User defined functions
import strats as strat_lib
from utils import tickers, strategy

# For type hinting
from typing import Tuple
from numpy import array as np_arr
from pandas import DataFrame as pandasDF

def main(ga_config: dict) -> dict:
    '''
    Main running function for the genetic algorithm

    Parameters
    ----------
    ga_config : dict
        Config params for the algorithm

    Returns
    -------
    best_strat : dict
        The optimised strategy parameters
    '''
    
    # Initialise all the parameters needed to start the evolution
    data, strats, fit_arr, strats_to_calc = init_ga(ga_config)
    
    # This gathers the number of strategies to change on each evolution
    perc_change = int((1-ga_config['keep perc'])*ga_config['num strats'])

    # Start the optimisation procedure
    for evl in range(0, ga_config['num evolutions']):
        
        # Calculate the fitness for the strategies
        fit_arr = get_fitness(data,
                              ga_config,
                              strats,
                              fit_arr,
                              strats_to_calc,
                              )
        
        # Rank the strategies, and select the strategies to change
        ranks = fit_arr[fit_arr[:, 1].argsort()]
        good_strats = ranks[perc_change:, 0].astype(np.int32)
        bad_strats = ranks[:perc_change, 0].astype(np.int32)
        
        # Split the bad strategies into 3 approx equal sets to make modifications
        splits = np.array_split(bad_strats, 3)
        
        # Replace bad strategies with random new ones
        for strat in splits[0]:
            strats[str(strat)] = get_random_strat(ga_config)
            
        # Add random perturbations to good strategies
        for strat in splits[1]:
            rand_strat = str(random.choice(good_strats))
            strats[str(strat)] = perturb_strat(strats[rand_strat].copy(),
                                               ga_config,
                                               )
            
        # Breed good strategies to make another
        for strat in splits[2]:
            strats[str(strat)] = breed_good_strats(good_strats,
                                                   strats,
                                                   ga_config,
                                                   )
            
        # Tell the optimiser which strats have been changed to calculate the
        # fitness function of. This saves time on recalculating the good strats
        strats_to_calc = bad_strats 
        
        # Print out evolution stats 
        print(f'\nEvolution {evl}')
        for count, strat in enumerate(np.flipud(good_strats[-5:])):
            print(str(count) + '. Strategy: ' +  str(strat) + 
                  ', ' + ga_config['fitness'] + ': ' + 
                  str(fit_arr[strat, 1])
                  )
        print('----------------------------------------------')
        
    # Print the final results and save to json
    best_strat = strats[str(good_strats[-1])]
    best_strat['ticker opt'] = data['ticker'].unique().tolist()
    print_and_save(best_strat,
                   ga_config)
        
    return best_strat

def out_sample_test(strat: dict,
                    ga_config: dict):
    '''
    Perform in, and out of sample testing for evaluating the outcome of the ga
    '''

    # Get tickers to perform the out of sample testing
    out_sample_tickers = tickers.get_tickers_exc_sample(ga_config['num tickers test'],
                                                        strat['ticker opt'],
                                                        )
    
    # In sample test
    print('\n----------------------------------------------')
    print('In sample testing results: ')
    ticker_test(strat['ticker opt'],
                strat,
                ga_config,
                )
    print('----------------------------------------------')
    
    # Out of sample test
    print('\n----------------------------------------------')
    print('Out of sample testing results: ')
    ticker_test(out_sample_tickers,
                strat,
                ga_config,
                )
    print('----------------------------------------------')
    
    return

def ticker_test(tickers: list,
                strat: dict,
                ga_config: dict):
    '''
    Given a set of tickers, run the buy/sell algorithm and produce average
    statistics for the set of tickers
    '''
    
    # Key performance metric storage lists
    win_rate = []
    avg_profit = []
    mean_hold = []
    max_hold = []
    num_trades = []
    
    for ticker in tickers:
        df = pd.read_csv(f'data/{ticker}.csv')
        df = strategy.add_strat_cols(df, strat, ga_config['strat'])        
        _, stats = strategy.run_strategy(df, strat, ga_config['strat'])
        
        win_rate.append(stats['win rate'])
        avg_profit.append(stats['avg profit'])
        mean_hold.append(stats['mean hold'])
        max_hold.append(stats['max hold'])
        num_trades.append(stats['number of trades'])
        
    # Print the summaries
    print('Average win rate: ', np.mean(win_rate))
    print('Lowest win rate: ', np.min(win_rate))
    print('Average profit: ', np.mean(avg_profit))
    print('Maximum hold time: ', np.max(max_hold))
    print('Mean hold time: ', np.mean(mean_hold))
    print('Mean number of trades: ', np.mean(num_trades))
    
    return

def print_and_save(strat: dict,
                   ga_config: dict):
    '''
    Print the optimal parameters and save to a pickle file
    '''
    
    print('Most optimal parameters are: ')
    for k, v in strat.items():
        if k != 'ticker opt':
            print(k + ': ' + str(v))
    
    with open('ga/strategies/' + ga_config['save name'] + '.pkl', 'wb') as f:
        pickle.dump(strat, f)
        
    return
    
def init_ga(ga_config: dict) -> Tuple[pandasDF, dict, np_arr, np_arr]:
    '''
    Initialise any parameters and data needed for the genetic algorithm

    Parameters
    ----------
    ga_config : dict
        Config controls for the genetic algorithm

    Returns
    -------
    data : pandasDF
        Price data for all tickers we are optimising
    strats : dict
        A random set of strategies
    fit_arr : np_arr
        An array to store the fitness values in
    strats_to_calc : np_arr
        An array to indicate which strategies to calculate the fitness for
    '''
    
    # Check if the filing structure is correct
    check_folder()
    
    # Get a random set of tickers, and load in the data we will optimise
    ticker_list = tickers.get_random_tickers(ga_config['num tickers'])
    data = get_price_data(ticker_list)
    
    # Initialise with a set of random strategies
    strats = {f'{n}': get_random_strat(ga_config) 
              for n in range(0, ga_config['num strats'])}
    
    # Initialise an empty array to store the fitness values in
    fit_arr = np.vstack((np.arange(0, ga_config['num strats']),
                         np.zeros(ga_config['num strats']),
                         ),
                        ).T
    
    # Initialise the array to determine which strategies to calculate the
    # fitness for. In this case its all of them, but in the algo we only need
    # to calculate the fresh strategies
    strats_to_calc = np.arange(0, ga_config['num strats'])
    
    return data, strats, fit_arr, strats_to_calc
    
def get_fitness(data: pandasDF,
                ga_config: dict,
                strats: dict,
                fit_arr: np_arr,
                strats_to_calc: np_arr) -> np_arr:
    '''
    Calculate the fitness function for each strategy defined in strats_to_calc.
    This runs the buy/sell algorithm on each ticker, and then uses the mean
    of all results to generate the fitness value for each strategy.

    Parameters
    ----------
    data : pandasDF
        Price data for all tickers we are optimising
    ga_config : dict
        Config controls for the genetic algo
    strats : dict
        All strategies generated by the genetic algo
    fit_arr : np_arr
        An array to store the fitness results for each strategy
    strats_to_calc : np_arr
        Which strategies to calculate the fitness value for

    Returns
    -------
    fit_arr : np_arr
        The recalculated fitness values for the new strategies
    '''
    
    for strat in strats_to_calc:
        
        # Initialise a list to store the result of this strategy
        res = []
        
        for ticker in data['ticker'].unique():
            
            # Add the strategy columns to the dataframe
            df_strat = data[data['ticker'] == ticker].reset_index().drop(columns = ['index'])
            df_strat = strategy.add_strat_cols(df_strat,
                                               strats[str(strat)],
                                               ga_config['strat'],
                                               )
            
            # Run the buy/sell algorithm and produce the statistics
            _, stats = strategy.run_strategy(df_strat,
                                             strats[str(strat)],
                                             ga_config['strat'],
                                             )
            
            # We want to strongly encourage the algorithm to not take any strat
            # which performes trades less than min_trades, this prevents some
            # curve fitting to very rare events
            if stats['number of trades'] > ga_config['min trades']:
                res.append(stats[ga_config['fitness']])
            else:
                res.append(-100)
            
        # Find the average result for this strategy
        fit_arr[strat, 1] = np.mean(res)
        
    return fit_arr
        
def get_price_data(tickers: list) -> pandasDF:
    '''
    Generate a dataframe which contains all the price data for each ticker

    Parameters
    ----------
    tickers : list
        List of tickers to optimise the strategy for

    Returns
    -------
    df : pandasDF
        A dataframe of all price data for each ticker.
    '''
    
    dfs = []
    for ticker in tickers:
        df = pd.read_csv(f'data/{ticker}.csv')
        df['ticker'] = ticker
        dfs.append(df)
        
    return pd.concat(dfs)


def breed_good_strats(good_strats: np_arr,
                      strats: dict,
                      ga_config: dict) -> dict:
    '''
    Taking parameters from good strategies, breed a new one.
    '''
    
    new_strat = {}
    for param in strats['0'].keys():
        rand_strat = random.choice(good_strats)
        new_strat[param] = strats[str(rand_strat)][param]
        
    return check_params(new_strat, ga_config)

def get_random_strat(ga_config: dict) -> dict:
    '''
    Get a randomly generated strategy
    '''
    if ga_config['strat'] == 'simple ma crossover':
        return strat_lib.ma_crossover.get_random_strat(ga_config)

def check_params(strat: dict,
                 ga_config: dict) -> dict:
    '''
    Check if the parameters for the strategy make sense, adjust if they dont
    '''
    if ga_config['strat'] == 'simple ma crossover':
        return strat_lib.ma_crossover.check_params(strat, ga_config)
    
def perturb_strat(strat: dict,
                  ga_config: dict) -> dict:
    '''
    Perturb the parameters of the strategy slightly to generate a new strategy
    '''
    if ga_config['strat'] == 'simple ma crossover':
        strat = strat_lib.ma_crossover.perturb_strat(strat, ga_config)
        return check_params(strat, ga_config)

def check_folder():
    '''
    Check if the ga strategies folder exists, if not, create it
    '''
    if not os.path.isdir('ga/strategies'):
        os.mkdir('ga/strategies')