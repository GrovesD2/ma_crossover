import numpy as np
import pandas as pd

from numpy import array as np_arr
from pandas import DataFrame as pandasDF

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main(nn_config: dict):
    '''
    Get the input data, train the nn, and save the model.

    Parameters
    ----------
    nn_config : dict
        The nn config opts

    Returns
    -------
    None
    '''
    
    data = get_nn_data(nn_config)
    model = get_model(nn_config, data)
    
    # Form the optimiser
    optim = Adam(learning_rate = nn_config['learn rate'],
                 decay = nn_config['decay rate'],
                 )
    
    # Compile and fit the model
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optim,
                  metrics = ['accuracy'])
    
    # Fit the model
    model.fit(data['x train'],
              data['y train'],
              epochs = nn_config['epochs'],
              batch_size = nn_config['batch size'],
              validation_split = nn_config['validation split'])
    
    # Save the model
    model.save('nn/models/' + nn_config['model save name'])
    
    # Evaluate performance 
    score = model.evaluate(data['x test'],
                           data['y test'],
                           verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Plot the confusion matrix
    test_pred = np.argmax(model.predict(data['x test']), axis = 1)
    cm = confusion_matrix(y_true = data['test'][:, -1].astype(int),
                          y_pred = test_pred)
    cm_scaled = cm/cm.astype(np.float).sum(axis = 0)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm_scaled)
    disp.plot()
    
    return 

def equalise_classes(df: pandasDF) -> pandasDF:
    '''
    To prevent there being a large discrepency in the number of each class,
    print the number of each examples in each class, and ask whether the number
    of classes should be equalised before feeding into the nn (this reduces
    the number of training points)

    Parameters
    ----------
    df : pandasDF
        The pre-processed data to go into the nn.

    Returns
    -------
    df : pandasDF
        A dataframe which has potentially been class-equalised
    '''
    
    # Find how many there are of each class
    num_each_class = df['labels'].value_counts().values
    
    # Print the summary
    print('Number of examples for each class: ')
    for n in range(0, len(num_each_class)):
        print('Class ' + str(n) + ': ' + str(num_each_class[n]))
    print('\nDo you wish to equalise the classes? (y/n)')
  
    opt = input()
    
    # If the equal classes are wanted, randomly select n examples from each class
    if opt == 'y':
        classes = df['labels'].unique()
        min_class_example = np.min(num_each_class)
        
        # For each class, take a random sample of examples s.t. all classes
        # have the same number of elements
        dfs = [df[df['labels'] == c].sample(min_class_example) 
               for c in classes]
        return pd.concat(dfs)

    else:
        return df
    
def get_nn_data(nn_config: dict) -> dict:
    '''
    From the data-file, get the train/test data for the nn.

    Parameters
    ----------
    nn_config: dict
        The config controls for the NN

    Returns
    -------
    nn_data: dict
        The data for training and testing
    '''
    
    # Dictionary to store the outputs
    nn_data = {}
    
    # Load in the data and drop columns not required for the nn
    df = pd.read_csv('nn/data/' + nn_config['strat name'] + '.csv')
    df = df.drop(columns = ['ticker', 'Date']).dropna()
    df = df.drop_duplicates()
    
    # Check if equal examples of each class is wanted
    df = equalise_classes(df)

    # Shuffle the data
    data = df.values
    np.random.shuffle(data)
    
    # Split training and test data
    split = int(nn_config['train perc']*(data.shape[0]))
    
    nn_data['train'] = data[0:split, :]
    nn_data['x train'] = nn_data['train'][:, 0:-1].astype(np.float32)
    nn_data['y train'] = to_categorical(nn_data['train'][:, -1].astype(int),
                                        nn_config['classes'])
    
    nn_data['test'] = data[split:, :]
    nn_data['x test'] = nn_data['test'][:, 0:-1].astype(np.float32)
    nn_data['y test'] = to_categorical(nn_data['test'][:, -1].astype(int),
                                       nn_config['classes'])
    
    # A 3D array is required for the lstm input, so we reshape to accomodate
    if nn_config['model type'] in ['lstm', 'bidirectional']:
        nn_data['x train'] = reshape_rnn(nn_data['x train'],
                                         max(nn_config['time lags']),
                                         )
        nn_data['x test'] = reshape_rnn(nn_data['x test'],
                                        max(nn_config['time lags']),
                                        )
    return nn_data

def reshape_rnn(x: np_arr,
                n_lags: range) -> np_arr:
    '''
    If an RNN-type network is wanted, reshape the input so that it is a 3D
    array of the form (sample, time series, feature).

    Parameters
    ----------
    x : np_arr
        The data to reshape.
    n_lags : int
        The number of time-lags used.

    Returns
    -------
    x_new : np_arr
        The reshaped x array for the RNN layers.
    '''
    
    # Calculate the number of features we have in the nn (assumes all features
    # are of the same length)
    num_feats = x.shape[1]//n_lags
    
    # Initialise the new x array with the correct size
    x_new = np.zeros((x.shape[0], n_lags, num_feats))
    
    # Populate this array through iteration
    for n in range(0, num_feats):
        x_new[:, :, n] = x[:, n*n_lags:(n+1)*n_lags]
    
    return x_new

def get_model(nn_config: dict,
              nn_data: dict):
    '''
    Obtain NN model (probably better written in OOP, kept it easy for now)

    Parameters
    ----------
    nn_config : dict
        Config controls for the nn
    nn_data : dict
        The data for training and testing

    Returns
    -------
    model : nn model
        The model architecture for training
    '''
    
    if nn_config['model type'] == 'vanilla':
        model = Sequential([
            Dense(nn_config['nodes'],
                  activation = 'relu',
                  input_shape = (nn_data['x train'].shape[1],)),
            Dense(nn_config['nodes']//2, activation = 'relu'),
            Dropout(nn_config['dropout perc']),
            Dense(nn_config['nodes']//4, activation = 'relu'),
            Dropout(nn_config['dropout perc']),
            Dense(nn_config['nodes']//8, activation = 'relu'),
            Dense(nn_config['classes'], activation = 'softmax')
            ])
        
    elif nn_config['model type'] == 'lstm':
        model = Sequential([
            LSTM(nn_config['nodes'],
                 input_shape = (nn_data['x train'].shape[1],
                                nn_data['x train'].shape[2]),
                 return_sequences = True),
            Dropout(nn_config['dropout perc']),
            LSTM(nn_config['nodes']),
            Dropout(nn_config['dropout perc']),
            Dense(nn_config['classes'], activation = 'softmax'),
            ])
        
    elif nn_config['model type'] == 'bidirectional':
        model = Sequential([
            Bidirectional(LSTM(nn_config['nodes'],
                               input_shape = (nn_data['x train'].shape[1],
                                              nn_data['x train'].shape[2]),
                               return_sequences = True),
                          ),
            Dropout(nn_config['dropout perc']),
            Bidirectional(LSTM(nn_config['nodes'])),
            Dropout(nn_config['dropout perc']),
            Dense(nn_config['classes'], activation = 'softmax'),
            ])
        
    return model