'''
Main function to train the neural network, given the config params
'''
from nn import funcs

if __name__ == "__main__":

    # Configuration settings for the Neural Network
    nn_config = {'save name': 'simple', # Model save name 
                 'tickers': 2, # How many tickers used to train the nn
                 'time lags': range(1, 50), # Which days back to include in the nn features
                 'train perc': 0.8, # Train/test split
                 'model name': 'vanilla', # Option to chose 'vanilla' or 'lstm'
                 'nodes': 512, # Number of nodes in layer 1
                 'dropout perc': 0.2, # Dropout percentage of any dropout layers
                 'epochs': 20, 
                 'batch size': 4, 
                 'learn rate': 1e-4,
                 'decay rate': 1e-5,
                 'validation split': 0.1}
    
    # Configuration settings for the buy/sell solver
    strat_config = {'slow type': 'rolling',
                    'slow price': 'Close',
                    'slow days': 50,
                    'fast type': 'rolling',
                    'fast price': 'Close',
                    'fast days': 10,
                    'profit': 10,
                    'stop': -5,
                    'max hold': 30}
    
    funcs.main(nn_config, strat_config)