
def get_config(strat: str) -> dict:
    '''
    Given a strategy name, return the strategy configurations

    Parameters
    ----------
    strat : str
        The name of the strategy to get the nn data for

    Returns
    -------
    config : dict
        The configuration parameters for the strategy
    '''
    
    if strat == 'simple ma crossover':
        return {'slow type': 'rolling',
                'slow price': 'Close',
                'slow days': 115,
                'fast type': 'rolling',
                'fast price': 'Open',
                'fast days': 114,
                'profit': 10,
                'stop': -10,
                'max hold': 10,
                
                'price feats': ['Open', 'Low', 'High', 'Close', 'fast_ma', 'slow_ma'],
                'other feats': ['Volume'],
                'drop cols': ['Adj Close']
                }
    
    elif strat == 'simple bollinger band':
        return {'mean type': 'exp',
                'std type': 'exp',
                'mean price': 'Open',
                'std price': 'Open',
                'mean days': 266,
                'std days': 246,
                'factor': -1.09,
                'profit': 10,
                'stop': -10,
                'max hold': 10,
                
                'price feats': ['Open', 'Low', 'High', 'Close', 'boll_lower'],
                'other feats': ['Volume'],
                'drop cols': ['Adj Close']
                }