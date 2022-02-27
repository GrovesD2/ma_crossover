
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
                'slow days': 30,
                'fast type': 'rolling',
                'fast price': 'Close',
                'fast days': 10,
                'profit': 5,
                'stop': -5,
                'max hold': 5,
                
                'price feats': ['Open', 'Low', 'High', 'Close', 'fast_ma', 'slow_ma'],
                'other feats': ['Volume'],
                'drop cols': ['Date', 'Adj Close']
                }
    
    elif strat == 'simple bollinger band':
        return {'mean type': 'rolling',
                'std type': 'exp',
                'mean price': 'Open',
                'std price': 'Open',
                'mean days': 86,
                'std days': 67,
                'factor': -0.2157809407816873,
                'profit': 10,
                'stop': -10,
                'max hold': 15,
                
                'price feats': ['Open', 'Low', 'High', 'Close', 'boll_lower'],
                'other feats': ['Volume'],
                'drop cols': ['Date', 'Adj Close']
                }