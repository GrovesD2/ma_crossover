import numpy as np
import pandas as pd

# Project imports
from utils import dates
#import dates

# Other imports and type-hinting
from functools import reduce
from pandas import DataFrame as pandasDF

def load_fundamental_data(ticker: str, 
                          period: str) -> pandasDF:
    '''
    Load in the fundamental data, given a ticker and a period. This loads the
    Cash Flow (CF), Balance Sheet (BS) and Income Statement (IS).

    Parameters
    ----------
    ticker : str
        The ticker to load the fundamental data for
    period : str
        The period to consider (either 'annual' or 'quarterly')

    Returns
    -------
    pandasDF
        The dataframe with all fundamental info in
    '''
    
    # Load in the cash-flow, balance sheet and income statement
    dfs = [pd.read_csv(f'data/{ticker}_{period}_{sheet}.csv')
           for sheet in ['CF', 'BS', 'IS']]
    
    # Merge all the dataframes such that all fundamentals are in one wide dataframe
    df = reduce(lambda left, right: pd.merge(left, right, 
                                             on = 'fiscalDateEnding',
                                             how = 'left'),
                dfs,
                )
    
    # Drop any duplicate columns which appear after the merge
    df = df.rename(columns = {col: col.split('_')[0] for col in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Sometimes the string 'None' appears, replace this with a nan value so it can
    # be processed numerically
    df = df.replace('None', value = np.nan)
    
    # Drop the currency column, and change all columns except the date to float
    df = df.drop(columns = 'reportedCurrency')
    df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric)
    
    # Sort the date to keep consistent between tickers
    return df.sort_values('fiscalDateEnding', ascending = False)

def get_f_score_df(df: pandasDF) -> pandasDF:
    '''
    Get the a binary dataframe indicating which criteria are filled in the
    Piotorski F-score for each date in the fundamental dataset
    
    Paramters
    ---------
    df : pandasDF
        A dataframe of stock fundamentals
    
    Returns
    -------
    df : pandasDF
        A dataframe containing each of the F-score components
    
    Notes
    -----
    Strategy taken from here:
    https://www.investopedia.com/terms/p/piotroski-score.asp
    
    Some calculations taken from here:
    https://www.quant-investing.com/blog/this-academic-can-help-you-make-better-investment-decisions-piotroski-f-score

    F1 = positive net income
    F2 = positive return on assets
    F3 = positive operating cash flow
    F4 = operating cash flow > net income
    F5 = lower long term debt in the current period to the last
    F6 = higher current ratio in the previous period as compared to the last
    F7 = no new shares in the current period
    F8 = higher gross margin in the current period
    F9 = higher asset turnover ratio
    '''
    
    # Parameters requried for the derivation of the F-score
    df['returnOnAssets'] = df['netIncome']/df['totalAssets']
    df['currentRatio'] = df['totalCurrentAssets']/df['totalCurrentLiabilities']
    df['newShares'] = df['commonStock'] - df['commonStock'].shift(-1)
    df['grossMargin'] = df['totalRevenue'] - df['costofGoodsAndServicesSold']
    df['assetTurnoverRatio'] = df['totalRevenue']/df['totalAssets']

    # Calculating each of the F-scores
    df['F1'] = np.where(df['netIncome'] > 0, 1, 0)
    df['F2'] = np.where(df['returnOnAssets'] > 0, 1, 0)
    df['F3'] = np.where(df['operatingCashflow'] > 0, 1, 0)
    df['F4'] = np.where(df['operatingCashflow'] > df['netIncome'], 1, 0)
    df['F5'] = np.where(df['longTermDebt'] < df['longTermDebt'].shift(-1), 1, 0)
    df['F6'] = np.where(df['currentRatio'] > df['currentRatio'].shift(-1), 1, 0)
    df['F7'] = np.where(df['newShares'] <= 0, 1, 0)
    df['F8'] = np.where(df['grossMargin'] > df['grossMargin'].shift(-1), 1, 0)
    df['F9'] = np.where(df['assetTurnoverRatio'] > df['assetTurnoverRatio'].shift(-1), 1, 0)

    # Drop the last row, since there is no previous period before that
    df = df[0:-1]

    # Return only the F-scores and the date, so it can be used for df merges
    return df[['fiscalDateEnding'] + [f'F{score}' for score in range(1, 10)]]

def get_fundamental_metrics(df: pandasDF) -> pandasDF:
    '''
    Obtain metrics/ratios from a stocks fundamental data.

    Parameters
    ----------
    df : pandasDF
        The fundamental stock data.

    Returns
    -------
    df : pandasDF
        The metrics/ratios from the fundamentals.
        
    Notes
    -----
    Any new metrics cannot have large values, since this will impact the
    training of the NN. Ideally, they need to be as close to 1 as possible.
    '''
    
    # Find the return on assets, which indicates how profitable a company is in
    # relation to its total assets
    df['returnOnAssets'] = df['netIncome']/df['totalAssets']
    
    # Find the ratio of shares from this period to the last, which indicates 
    # whether there is any dilution in the value
    df['sharesRatio'] = df['commonStock']/df['commonStock'].shift(-1)
    
    # This measures if the company has enough cash and short-term assets on hand
    # to pay bills
    df['currentRatio'] = df['totalCurrentAssets']/df['totalCurrentLiabilities']
    
    # This ratio is a quick way to see if a company is able to meet its short
    # term commitments with current, short-term liquid assets
    df['quickRatio'] = (df['totalCurrentAssets'] - df['inventory'])/df['totalCurrentLiabilities']
    
    # This measures a companies efficiency of generating revenue or sales
    df['assetTurnoverRatio'] = df['totalRevenue']/df['totalAssets']
    
    # These determines how dependent a business is on debt
    df['debtEquity'] = df['totalCurrentLiabilities']/df['totalShareholderEquity']
    df['longDebtEquity'] = df['longTermDebt']/df['totalShareholderEquity']
    df['shortDebtEquity'] = df['shortTermDebt']/df['totalShareholderEquity']
    
    # Truncate to only the metrics calculated in this function
    return df[['fiscalDateEnding', 'returnOnAssets', 'sharesRatio',
               'currentRatio', 'quickRatio', 'assetTurnoverRatio', 
               'debtEquity', 'longDebtEquity', 'shortDebtEquity']].dropna()

def get_all_metrics(ticker: str,
                    period: str) -> pandasDF:
    '''
    Get all the fundamental metrics and F-scores.

    Parameters
    ----------
    ticker : str
        The ticker to load the fundamental data for
    period : str
        Whether to consider 'annual' or 'quarterly'

    Returns
    -------
    df : pandasDF
        A dataframe of stock fundamental metrics
    '''
    
    # Load in the fundamental data
    df = load_fundamental_data(ticker, period)

    # Get fundamental metric dataframes
    F_score = get_f_score_df(df.copy())
    metrics = get_fundamental_metrics(df.copy())
    
    # Join these dataframes into one fundamental dataset
    return metrics.merge(F_score,
                         on = 'fiscalDateEnding',
                         how = 'left',
                         )

def get_all_fundamentals(ticker: str) -> pandasDF:
    '''
    Get both the quarterly and annual fundamentals metrics and F-scores in one
    dataframe.

    Parameters
    ----------
    ticker : str
        The ticker to load the data for

    Returns
    -------
    fund : pandasDF
        The dataframe containing the quarterly and yearly fundamental metrics
    '''
    
    # Get quarterly data, and adjust such that there is a date range in which
    # this fundamental data applies for
    quarterly = get_all_metrics(ticker, 'quarterly')
    
    # Apply the same process for the annual data
    yearly = get_all_metrics(ticker, 'annual')
    
    # Merge the datasets together to make a full dataframe
    fund = dates.backwards_date_merge(quarterly,
                                      yearly,
                                      'fiscalDateEnding',
                                      'fiscalDateEnding',
                                      'date')
    
    # Eliminate the existing date column from the yearly, which is no longer
    # required since other merges can be done on the quarterly date.
    fund = fund.drop(columns = 'fiscalDateEnding_y')
    
    # Rename the other date to an easier name to join on later
    fund = fund.rename(columns = {'fiscalDateEnding_x': 'fiscal_quarter'})
    
    # Find all numerical columns for filtering outliers
    cols = fund.select_dtypes(include=np.number).columns.tolist()
    
    # Filter any outliers that appear in the df
    for col in cols:
        fund = fund[(fund[col] < 50) & (fund[col] > -50)]
    
    return fund