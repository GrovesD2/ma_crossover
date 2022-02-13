'''
Main function to generate the input data to the NN
'''
from nn import gen_input

if __name__ == "__main__":

    # Configuration settings for the NN data
    nn_config = {'save name': 'simple', # Data save name 
                 'tickers': 300, # How many tickers used to train the nn
                 'time lags': range(1, 26), # Which days back to include in the nn features
                 'price feats': ['Open', 'Low', 'High', 'Close', 'fast_ma', 'slow_ma'],
                 'other feats': ['Volume'],
                 'drop cols': ['Date', 'Adj Close'],
                 }
    
    # Configuration settings for the buy/sell solver
    strat_config = {'slow type': 'rolling',
                    'slow price': 'Close',
                    'slow days': 30,
                    'fast type': 'rolling',
                    'fast price': 'Close',
                    'fast days': 10,
                    'profit': 5,
                    'stop': -5,
                    'max hold': 5}
    
    gen_input.main(nn_config, strat_config)