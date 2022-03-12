'''
Main function to generate the input data to the NN
'''
from nn import gen_input, strat_configs

if __name__ == "__main__":

    # Configuration settings for the NN data
    nn_config = {'strat name': 'simple ma crossover',
                 'tickers': 300, # How many tickers used to train the nn
                 'time lags': range(1, 26), # Which days back to include in the nn features
                 }

    # Generate the input for the NN
    gen_input.main(nn_config,
                   strat_configs.get_config(nn_config['strat name']),
                   )