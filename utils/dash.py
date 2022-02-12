'''
Functions for the dashboard
'''
import plotly.graph_objects as go

from typing import Tuple
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots

from pandas import DataFrame as pandasDF

def pm_months(date: str,
              months: int) -> list:
    '''
    Find the date plus and minus one month from the input date

    Parameters
    ----------
    date : str
        The date which we are interested in.

    Returns
    -------
    date_range : list
        The range [date - 1 month, date + 1 month], the entries are strings
    '''
    # Change the string representation to a datetime object
    date = dt.strptime(date, '%Y-%m-%d')
    
    # Add and subtract a month from the date to give the range to plot over,
    # change this to a string representation for filtering
    date_range = [date + relativedelta(months = m) for m in [-months, months]]
    return [dt.strftime(d, '%Y-%m-%d') for d in date_range]

def get_sold_price(df: pandasDF,
                   bought_date: str,
                   profit: float) -> float:
    '''
    Get the profit/loss made from a trade

    Parameters
    ----------
    df : pandasDF
        The price data
    bought_date : str
        The date at which the stock was bought.
    profit : float
        The profit made from this trade.

    Returns
    -------
    sold_price : float
        The price the stock was sold for.
    '''
    return df[df['Date'] == bought_date]['Open'].values[0]*(1+profit/100)

def get_plotting_info(df: pandasDF,
                      df_summary: pandasDF,
                      plot_idx: int) -> Tuple[pandasDF, dict, dict]:
    '''
    

    Parameters
    ----------
    df : pandasDF
        The price data
    df_summary : pandasDF
        The summary of trades made for this strategy
    plot_idx : int
        The index of the trade we wish to plot

    Returns
    -------
    df : pandasDF
        A dataframe of prices filtered to a pm 2 month window of buying
    bought : dict
        Information on when the stock was bought, and at what price.
    sold : dict
        Information on when the stock was sold, and at what price.
        
    Notes
    -----
    The bought/sold date/price needs to be in a list/array. Plotly does not 
    deal with single float values.
    '''
    
    # Get the bought date 
    df_summary = df_summary.loc[[plot_idx]]
    bought_date = df_summary['Bought'].tolist()[0]
    
    # Get the date range to plot the candlesticks over
    date_range = pm_months(bought_date, 2)
    df = df[(df['Date'] >= date_range[0]) & 
            (df['Date'] <= date_range[1])]
    
    # Get the value the stock was bought at
    bought =  {'Date': [bought_date],
               'Price': [df[df['Date'] == bought_date]['Open'].values[0]]}
    
    # Get the value the stock was sold at
    sold =  {'Date': [df_summary['Sold'].tolist()[0]],
             'Price': [get_sold_price(df,
                                      bought_date,
                                      df_summary['Profit/Loss'].tolist()[0])]}

    return df, bought, sold

def get_candlestick_plot(df: pandasDF,
                         bought: dict,
                         sold: dict):
    '''
    This func provides the candlestick chart for a single trade to display on
    the dashboard.

    Parameters
    ----------
    df : pandasDF
        A filtered dataframe, displaying the price info pm 2 months from the
        bought point
    bought : dict
        Information on when the stock was bought, and at what price.
    sold : dict
        Information on when the stock was sold, and at what price.

    Returns
    -------
    fig : plotly figure
        The candlestick chart to be included on the dashboard
    '''

    fig = make_subplots(rows = 2,
                        cols = 1,
                        shared_xaxes = True,
                        vertical_spacing = 0.1,
                        subplot_titles = ('Candlestick Chart', 'Volume'),
                        row_width = [0.2, 0.7])
    
    # Plot the OLHC candlestick chart
    fig.add_trace(go.Candlestick(x = df['Date'],
                                 open = df['Open'],
                                 high = df['High'],
                                 low = df['Low'],
                                 close = df['Close'],
                                 name = 'OHLC',
                                 ), 
                  row=1,
                  col=1,
                  )
    
    # Add the rolling averages
    fig.add_trace(go.Scatter(x = df['Date'],
                             y = df['slow_ma'],
                             name = 'slow ma',
                             ),
                  row=1,
                  col=1,
                  )
    fig.add_trace(go.Scatter(x = df['Date'],
                             y = df['fast_ma'],
                             name = 'fast ma',
                             ),
                  row=1,
                  col=1,
                  )
    
    # Add the buy and sell points
    fig.add_trace(go.Scatter(x = bought['Date'],
                             y = bought['Price'],
                             name = 'Bought',
                             marker = {'size': 10}
                             ),
                  row=1,
                  col=1,
                  )
    fig.add_trace(go.Scatter(x = sold['Date'],
                             y = sold['Price'],
                             name = 'Sold',
                             marker = {'size': 10}
                             ),
                  row=1,
                  col=1,
                  )
    
    # Volume plot on the second row
    fig.add_trace(go.Bar(x = df['Date'],
                         y=df['Volume'],
                         name = 'Volume'),
                  row=2,
                  col=1,
                  )
    
    # Do not show OHLC's rangeslider plot 
    fig.update(layout_xaxis_rangeslider_visible=False)
        
    return fig