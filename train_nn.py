'''
Main function to train the NN
'''

from nn import model, testing

if __name__ == "__main__":  

    # Configuration settings for the Neural Network
    nn_config = {'strat name': 'simple bollinger band', # Name of the strategy
                 'model save name': 'simple bb',
                 'time lags': range(1, 26), # Which days back to include in the nn features
                 'train perc': 0.8, # Train/test split
                 'model type': 'vanilla', # Option to chose 'vanilla', 'lstm', or 'bidirectional'
                 'classes': 4, 
                 'nodes': 256, # Number of nodes in layer 1
                 'dropout perc': 0.4, # Dropout percentage of any dropout layers
                 'epochs': 25, 
                 'batch size': 4, 
                 'learn rate': 1e-3,
                 'decay rate': 1e-4,
                 'validation split': 0.1,
                 
                 # Number of tickers to perform the out of sample test on
                 'num tickers test': 100,
                 
                 # Minimum class level to use in the testing
                 'min class': 3,
                 
                 # Surety level threshold for the testing. The NN will not
                 # consider any predictions where it is less than x% sure
                 'surety': 0.5,
                 }
    
    # Train the nn model
    model.main(nn_config)
    
    # Out of sample test the NN model
    testing.main(nn_config)