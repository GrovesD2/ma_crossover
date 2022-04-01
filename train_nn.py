'''
Main function to train the NN
'''

from nn import model, testing

if __name__ == "__main__":  

    # Configuration settings for the Neural Network
    nn_config = {'strat name': 'simple ma crossover', # Name of the strategy
                 'model save name': 'simple ma',
                 
                 # NN specific configs
                 'train date': '2019-01-01', # Train up to this date
                 'model type': 'bidirectional', # Option to chose 'vanilla', 'lstm', or 'bidirectional'
                 'classes': 2, 
                 'nodes': 128, # Number of nodes in layer 1
                 'dropout perc': 0.4, # Dropout percentage of any dropout layers
                 'epochs': 10, 
                 'batch size': 128, 
                 'learn rate': 1e-4,
                 'decay rate': 1e-4,
                 'regularise': 1e-3,
                 'validation split': 0.1,
                 
                 'out sample test': True,
                 
                 # Minimum class level to use in the testing
                 'min class': 1,
                 
                 # Surety level threshold for the testing. The NN will not
                 # consider any predictions where it is less than x% sure
                 'surety': 0.65
                 }
    
    # Train the nn model
    model.main(nn_config)
    
    if nn_config['out sample test']:
        # Out of sample test the NN model
        testing.main(nn_config)