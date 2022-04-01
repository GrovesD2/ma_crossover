'''
Main function to generate the input data to the NN
'''
from nn import gen_input, strat_configs

if __name__ == "__main__":

    # Configuration settings for the NN data
    nn_config = {'strat name': 'simple ma crossover',
                 'include fundamentals': False,
                 'tickers': 350, # How many tickers worth of data to use
                 'testing tickers': 150, # How many tickers to test on
                 
                 # Earliest date to consider in the training, this only applies
                 # for when fundamentals are not used
                 'lower date filter': '2008-01-01', 
                 
                 # The days back in time to include as features in the NN
                 'time lags': range(1, 41), # Which days back to include in the nn features
                 }

    # Generate the input for the NN
    gen_input.main(nn_config,
                   strat_configs.get_config(nn_config['strat name']),
                   )