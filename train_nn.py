'''
Main function to train the NN
'''

from nn import model

if __name__ == "__main__":  

    # Configuration settings for the Neural Network
    nn_config = {'data name': 'simple', # Name of the generated input data
                 'model save name': 'simple',
                 'time lags': range(1, 26), # Which days back to include in the nn features
                 'train perc': 0.8, # Train/test split
                 'model type': 'bidirectional', # Option to chose 'vanilla', 'lstm', or 'bidirectional'
                 'classes': 3, 
                 'nodes': 64, # Number of nodes in layer 1
                 'dropout perc': 0.2, # Dropout percentage of any dropout layers
                 'epochs': 15, 
                 'batch size': 8, 
                 'learn rate': 1e-4,
                 'decay rate': 1e-5,
                 'validation split': 0.1}
    
    model.main(nn_config)