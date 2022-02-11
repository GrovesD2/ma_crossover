'''
This file is temporary! Only for use in developing the running of the stratgey
and the creation of the dashboard.
'''

import pandas as pd

from utils import strategy

# Configuration settings for the buy/sell solver
config = {'slow type': 'rolling',
          'slow price': 'Close',
          'slow days': 50,
          'fast type': 'rolling',
          'fast price': 'Close',
          'fast days': 10,
          'profit': 10,
          'stop': -5,
          'max hold': 30}

# Read in the csv file for this study
df = pd.read_csv('data/AAPL.csv')

# Run the strategy functions to get the results
df = strategy.add_strat_cols(df, config)
df_summary, stats = strategy.run_strategy(df, config)