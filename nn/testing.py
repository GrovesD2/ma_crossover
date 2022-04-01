import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# For type hinting
from typing import Tuple
from numpy import array as np_arr
from pandas import DataFrame as pandasDF

# Project imports
from nn import strat_configs, model, utils
    
def main(config):
    '''
    Perform the out-of-sample testing for the nn. In this testing the nn acts
    as an advisor, meanining that it predicts if the strategy buy signal is
    true or false.

    Parameters
    ----------
    config : dict
        Config params for the out of sample test.

    Returns
    -------
    None
    '''
    
    # Adjust the config file with the saved settings from the data generation
    config = utils.add_gen_config(config)
    
    # Get the strategy config for what the nn was trained on
    strat_config = strat_configs.get_config(config['strat name'])
    
    # Get the testing data
    df_test, data, true_labels = get_testing_data(config,
                                                  strat_config,
                                                  )
    
    # Load in the nn model for the predictions
    nn_model = load_model('nn/models/' + config['model save name'])
    
    # Make the predictions and find the class labels
    predict = nn_model.predict(data.astype('float32'))
    labels = np.argmax(predict, axis = 1)
    
    # Make a new df to signify the return
    df_test['predicted'] = labels
    df_test['surety'] = predict[np.arange(0, labels.shape[0]), labels]
    
    # Save the testing to a csv file so that future MC trials can be performed
    df_test.to_csv('nn/data/' + config['strat name'] + ' out sample test.csv',
                   index = False)
    
    # Plot the confusion matrix
    utils.get_confusion_matrix(true_labels,
                               labels,
                               config,
                               'Out of sample test'
                               )

    return get_stats(df_test, config)
    
def get_testing_data(config: dict,
                     strat_config: dict) -> Tuple[pandasDF, np_arr, np_arr]:
    '''
    Get the input numpy array for the dat

    Parameters
    ----------
    config : dict
        Config params for the out of sample test.
    strat_config : dict
        Config params for the strategy the nn was trained on

    Returns
    -------
    df : pandasDF
        The testing dataframe, filtered to only the ticker/profit/label columns
    data : np_arr
        The data to feed into the NN
    labels: np_arr
        The true labels for the confusion matrix
    '''
    
    # Load in the dataframe with the tickers to test with
    df = pd.read_csv('nn/data/' + config['strat name'] + ' testing.csv')
    
    # Reset the index of the dataframe to remove the set with copy warning
    df = df.reset_index().drop(columns = ['index', 'ticker', 'Date'])
    
    # Change the df to values, and reshape if an LSTM-type network has been used
    data = df.values[:, :-2]
    if config['model type'] in ['lstm', 'bidirectional']:
        data = model.reshape_rnn(data,
                                 max(config['time lags']),
                                 )
    
    return df[['Profit/Loss', 'labels']], data, df['labels'].values

def get_stats(df: pandasDF,
              config: dict):
    '''
    Get the statistics of how the strategy performs with and without the NN

    Parameters
    ----------
    df : pandasDF
        A dataframe of trades made for this strategy
    config : dict
        Config params for the out of sample test

    Returns
    -------
    None
    '''
  
    # First, deduce how the strategy performed without the NN to assist
    print('\nPerformance without the NN')
    print_stats(df['Profit/Loss'].values)
    
    # Filter the dataframe to only consider the minimum class and surety level
    df = df[(df['predicted'] >= config['min class'])
            & (df['surety'] > config['surety'])]
    
    # Now determine how the strategy performed with the NN
    print('\nPerformance with the NN')
    print_stats(df['Profit/Loss'].values)
    
    return

def print_stats(percs: np_arr):
    '''
    Given an array of profit/loss percentages, print the statistics.

    Parameters
    ----------
    percs : np_arr
        The profit/loss from each trade

    Returns
    -------
    None
    '''
    
    if percs.shape[0] > 0:
        print('Number of trades: ', percs.shape[0])
        print('Win rate: ', percs[percs > 0].shape[0]/percs.shape[0])
        print('Mean profit: ', np.mean(percs))
        print('Median profit: ', np.median(percs))
    else:
        print('No trades satisfied the class label/surety combination')
    
    return