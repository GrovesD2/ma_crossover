'''
Functions for the neural network
'''

import os
import numba as nb
import pandas as pd

# Project imports
from utils import tickers, strategy, dates, fundamentals

# Other imports and type-hinting
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
    
    # Check if the data-folder has been made
    check_folder()
    
    # Get a list of tickers, based on the available data
    ticker_list = tickers.get_random_tickers(nn_config['tickers'])
    
    # Generate the nn input
    df = get_nn_input(ticker_list,
                      nn_config,
                      strat_config,
                      )
    
    # Drop the profit/loss column, this is required to make the gen_nn_input 
    # function multi-purpose between training and testing
    df = df.drop(columns = ['Profit/Loss'])

    # Save the input data into the data-folder
    df.to_csv('nn/data/' + nn_config['strat name'] + '.csv',
              index = False)
    
    return 

def get_nn_input(tickers: list,
                 nn_config: dict,
                 strat_config: dict) -> pandasDF:
    '''
    Get the dataframe of all inputs to the nn

    Parameters
    ----------
    tickers : list
        A list of tickers to generate the inputs for
    nn_config : dict
        Configuration parameters for the nn
    strat_config : dict
        Configuration parameters for the solver

    Returns
    -------
    df : pandasDF
        The processed inputs for the nn
    '''

    # Get the backseries of data for each trade per each ticker
    df = get_time_series_data(nn_config,
                              strat_config,
                              tickers)

    # Drop the columns we no longer want for the nn
    df = df.drop(columns = (strat_config['price feats']
                            + strat_config['other feats']
                            + strat_config['drop cols']
                            ),
                 )
    
    # Normalise the time-series for price related cols, and other features
    df = normalise_prices(df, strat_config['price feats'])
    df = normalise_non_price(df, strat_config['other feats'])
    
    # Get the spy index as a feature
    df = single_stock_data('SPY', nn_config).merge(df,
                                                   on = 'Date',
                                                   how = 'inner',
                                                   )
    
    if nn_config['include fundamentals']:
        df = include_fundamentals(df)
        
        # Drop the fiscal quarter column since it is no longer required
        df = df.drop(columns = 'fiscal_quarter')
        
        # Rearrange the columns such that ticker, Profit/Loss and labels appear
        # at the end, this is helpful for sorting the data in the nn code
        cols = [col for col in df.columns.tolist() 
                if col not in ['Profit/Loss', 'labels', 'ticker']]
        df = df[cols + ['Profit/Loss', 'labels', 'ticker']]
    
    # Drop any null entries, to stop any bad-data getting into the nn
    return df.dropna()

def include_fundamentals(df: pandasDF):
    
    dfs = []
    
    for ticker in df['ticker'].unique():
        
        df_tmp = df[df['ticker'] == ticker]
        try:
            fund = fundamentals.get_all_fundamentals(ticker)
            df_tmp = dates.backwards_date_merge(df_tmp,
                                                fund,
                                                'Date',
                                                'fiscal_quarter',
                                                'date')
            dfs.append(df_tmp)
        except Exception as e:
            print(ticker, ' : ', e)
            
    return pd.concat(dfs)
            

def single_stock_data(ticker: str,
                      nn_config: dict) -> pandasDF:
    '''
    Get a single stock's time-series history as a nn feature. I.e. one could
    use the spy as a feature to the NN.

    Parameters
    ----------
    ticker : str
        The ticker to get the time-series data for
    nn_config : dict
        The config params for the nn.

    Returns
    -------
    df : pandasDF
        The time series data for a single ticker.
    '''
    
    df = pd.read_csv(f'data/{ticker}.csv')

    # Add the time-series columns, and then normalise them
    for col in ['Open', 'Low', 'High', 'Close', 'Volume']:
        df = time_series_cols(df, col, nn_config['time lags'])
    
    df = normalise_prices(df, ['Open', 'Low', 'High', 'Close'])
    df = normalise_non_price(df, ['Volume'])
    
    # Drop the unncessary columns when returning    
    return df.drop(columns = ['Open', 'Low', 'High', 'Close', 'Volume',
                              'Adj Close']
                   )
    
def normalise_prices(df: pandasDF,
                     price_cols: list):
    '''
    Normalise all price related time-series on the same scale.
    '''
    # Find all columns which are price related
    cols = []
    for feat in price_cols:
        cols += get_lagged_col_names(df, feat)
        
    # Normalise these cols agaisnt the max value in each row
    return normalise_time_series(df, cols)
        
def normalise_non_price(df: pandasDF,
                        other_feats: list):
    '''
    For each non-price related time-series, normalise them
    '''
    for feat in other_feats:
        cols = get_lagged_col_names(df, feat)
        df = normalise_time_series(df, cols)
        
    return df
    
def get_lagged_col_names(df: pandasDF,
                         feat: str) -> list:
    '''
    Get all lagged column names for the specific feature
    '''
    return [col for col in df.columns.tolist() if feat in col]

def normalise_time_series(df: pandasDF,
                          cols: list) -> pandasDF:
    '''
    For a set of columns normalise by using a standard scaling approach
    '''
    df[cols] = df[cols].apply(lambda x: (x-x.mean())/x.std(ddof = 0),
                              axis = 1,
                              )
    return df

def get_time_series_data(nn_config: dict,
                         strat_config: dict,
                         ticker_list: list) -> pandasDF:
    '''
    Produce a dataframe containing the time-series data for a random selection
    of tickers. The output df contains the backseries of data for each trade
    made in the buying and selling algorithm.

    Parameters
    ----------
    nn_config : dict
        The config params for the nn
    strat_config : dict
        The config params for the trading strategy
    ticker_list : list
        A list of tickers to run the buy/sell for, and produce input data for
        the nn

    Returns
    -------
    dfs : list
        A dataframe containing all trades
    '''    
    # Open an empty list to store the processed dataframes per each ticker
    dfs = []

    # For each ticker in the ticker list, run the buy/sell and pre-process
    # the data for input into the nn
    for ticker in ticker_list:
        
        df = pd.read_csv(f'data/{ticker}.csv')
        df = strategy.add_strat_cols(df,
                                     strat_config,
                                     nn_config['strat name'],
                                     )
        
        # Since we only have fundamental data since 2017, there is no point
        # searching for training data beyond that
        if nn_config['include fundamentals']:
            df = df[df['Date'] >= '2017-01-01']
        else:
            df = df[df['Date'] >= '2010-01-01']
        
        if len(df) > 0:
            df_strat, _ = strategy.run_strategy(df,
                                                strat_config,
                                                nn_config['strat name'],
                                                )
            
            # If at least one trade has been made, create the nn features
            if len(df_strat) > 0:
                
                # Obtain the time-series of data for each of the features
                for col in strat_config['price feats'] + strat_config['other feats']:
                    df = time_series_cols(df, col, nn_config['time lags'])
                    
                # Filter the dataframe to consider the times a trade was made
                df = get_trades(df, df_strat)
                
                # Produce the tags
                df['labels'] = labeller(df['Profit/Loss'].values,
                                        strat_config['profit'],
                                        strat_config['stop'])
                
                # Add the identifier for this data, and include it on the list
                df['ticker'] = ticker
                dfs.append(df)

    return pd.concat(dfs)
                             
def get_trades(df: pandasDF,
               df_strat: pandasDF) -> pandasDF:
    '''
    Merge the time-series dataframe with the strategy dataframe, to get nn
    inputs that correspond to the trades made (and not other points in time)

    Parameters
    ----------
    df : pandasDF
        The price dataframe with the time-series cols
    df_strat : pandasDF
        The summary of the trades made for this ticker

    Returns
    -------
    df : pandasDF
        The filtered df which contains the lagged time-series for the trades
    '''
    df_strat = df_strat.rename(columns = {'Bought': 'Date'})
    df = df.merge(df_strat, on = 'Date', how = 'inner')
    
    # Drop the unecessary columns for the nn
    return df.drop(columns = ['Sold' , 'Days held'])
        
def time_series_cols(df: pandasDF,
                     col: str,
                     lags: list) -> pandasDF:
    '''
    Create columns for the time series of "cols", where the lags are specified
    in the "lags" list.

    Parameters
    ----------
    df : pandasDF
        The dataframe of price data.
    col : str
        The column we wish to create the time series of.
    lags : list
        A list of integers to denote the time series points.

    Returns
    -------
    pandasDF
        A dataframe with the time series cols.
    '''
    return df.assign(**{
        f'{col}_t-{lag}': df[col].shift(lag)
        for lag in lags
        })

@nb.jit(nopython=True) 
def labeller(percs, profit, stop):
    
    out = []
    for perc in percs:
        if int(perc) == profit:
            out.append(3)
        elif int(perc) == stop:
            out.append(0)
        elif perc > 0:
            out.append(2)
        else:
            out.append(1)
    
    return out

def check_folder():
    '''
    Check if the nn data/models folder exists, if they do not, create them.
    '''
    if not os.path.isdir('nn/data'):
        os.mkdir('nn/data')
        
    if not os.path.isdir('nn/models'):
        os.mkdir('nn/models')