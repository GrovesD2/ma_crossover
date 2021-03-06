'''
Functionality to get ticker names and data
'''
import os
import random
import pandas as pd
import yfinance as yf

def get_random_tickers(n: int) -> list:
    '''
    From the available ticker data, randomly select n tickers.
    '''
    return random.sample(get_all_ticker_names(), n)

def get_tickers_exc_sample(n: int,
                           exclude: list) -> list:
    '''
    Get a random selection of tickers from all available, excluding any tickers
    that are in the exclude list.
    '''
    all_tickers = get_all_ticker_names()
    ticker_sample = [ticker for ticker in all_tickers if ticker not in exclude]
    return random.sample(ticker_sample, n)

def get_all_ticker_names():
    '''
    Get a list of all ticker names in the data-directory.
    '''
    # To remove duplicated entries since fundamentals exist in this folder too,
    # the list is turned into a set and back to a list again
    return list(
        set([s.split('.csv')[0].split('_')[0]
             for s in os.listdir('data/') if '.csv' in s]
            )
        )

def check_run(download_type: str):
    '''
    Before downloading tickers, check the paths and configs are correct

    Parameters
    ----------
    download_type : str
        A config to describe which type of download is being performed
        
    Returns
    -------
    None

    '''
    # First check if the data folder exists, if it doesn't, then create one
    if not os.path.isdir('data/'):
        os.mkdir('data')
        
    # Now check the download type is of the correct format
    if download_type not in ['spy', 'user']:
        raise ValueError(str(download_type) + ' is not a valid option, ' + 
                         ' please enter spy or user.')

def get_tickers(download_type: str) -> list:
    '''
    Get the list of tickers based on the config controls

    Parameters
    ----------
    download_type : str
        Which data source to grab the ticker names from

    Returns
    -------
    tickers: list
        A list of tickers ready for downloading
    '''    
    if download_type == 'user':
        return user_defined()
    elif download_type == 'spy':
        return spy_tickers()
    
def user_defined() -> list:
    '''
    Get a list of tickers generated from the user file "tickers.csv"
    
    Parameters
    ----------
    None

    Returns
    -------
    tickers: list
        A list of tickers to download.
    '''
    return pd.read_csv('utils/tickers.csv')['ticker'].tolist()
    
def spy_tickers() -> list:
    '''
    Get the current s&p500 from wikipedia.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    tickers : list
        List of the current s&p 500 tickers
    '''
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    
    return df[0]['Symbol'].tolist()

def get_data(tickers: list):
    '''
    Obtain historical daily price data for all tickers specified. The outcome
    is a csv file of price data for each ticker in the data folder.

    Parameters
    ----------
    tickers : list
        A list of the tickers to download the data for

    Returns
    -------
    None
    '''
    
    print('Downloading the data from yahoo finance')
    data = yf.download(tickers = tickers,
                       interval = '1D',
                       group_by = 'ticker',
                       auto_adjust = False,
                       prepost = False,
                       threads = True,
                       proxy = None
                       )
    
    data = data.T
    
    print('Data downloaded. Saving the csv files in the data directory.')
    for ticker in tickers:
        
        # Try statement is required because sometimes a ticker fails to download
        try:
            df = data.loc[(ticker.upper(),),].T.reset_index().dropna()
            df.to_csv(f'data/{ticker}.csv', index = False)
        except:
            print(f'Ticker {ticker} failed to download.')