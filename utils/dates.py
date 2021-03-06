'''
All functionality relating to date conversions
'''

import pandas as pd

# Other imports and type-hinting
from pandas import DataFrame as pandasDF

def backwards_date_merge(df1: pandasDF,
                         df2: pandasDF,
                         col1: str,
                         col2: str,
                         alias: str) -> pandasDF:
    '''
    Perform an asof merge such that df1 looks backwards to find the nearest
    date in df2.

    Parameters
    ----------
    df1 : pandasDF
        Left dataframe for the join
    df2 : pandasDF
        Right dataframe for the join
    col1 : str
        The date column name in df1
    col2 : str
        The date column name in df2
    alias : str
        What to name the new, common date column as (dropped at the end)

    Returns
    -------
    df : pandasDF
        The backwards merged dataframe
    '''
    
    # Form the same column for the merge
    df1[alias] = pd.to_datetime(df1[col1])
    df2[alias] = pd.to_datetime(df2[col2])
    
    # The asof merge requires the date to be sorted
    df1 = df1.sort_values(alias)
    df2 = df2.sort_values(alias)
    
    # Perform the asof merge
    df = pd.merge_asof(df1, df2, on = alias, direction = 'backward')
    
    return df.dropna().drop(columns = alias)