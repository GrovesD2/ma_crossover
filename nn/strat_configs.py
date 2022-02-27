
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
        return {'mean type': 'exp',
                'std type': 'exp',
                'mean price': 'Open',
                'std price': 'Close',
                'mean days': 28,
                'std days': 156,
                'factor': -1.05,
                'profit': 3,
                'stop': -2,
                'max hold': 5,
                
                'price feats': ['Open', 'Low', 'High', 'Close', 'boll_lower'],
                'other feats': ['Volume'],
                'drop cols': ['Date', 'Adj Close']
                }