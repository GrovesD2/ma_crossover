'''
Functionality for saving/loading objects
'''

import pickle

def save_dict(d: dict,
              name: str):
    '''
    Save a dictionary to a pickle object

    Parameters
    ----------
    d : dict
        The dictionary to save as a pickle object.
    name : str
        The path and/or name for the pickle object

    Returns
    -------
    None
    '''
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(d, f)
    return

def load_dict(name: str):
    '''
    Load a dictionary stored as a pickle object

    Parameters
    ----------
    name : str
        The path and/or name for the pickle object

    Returns
    -------
    None
    '''
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)