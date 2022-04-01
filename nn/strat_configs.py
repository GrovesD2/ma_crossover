
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
                'slow days': 34,
                'fast type': 'rolling',
                'fast price': 'Open',
                'fast days': 31,
                'profit': 12.3,
                'stop': -7,
                'max hold': 3,
                
                'return feats': ['Open', 'Low', 'High', 'Close', 'fast_ma',
                                 'slow_ma', 'Volume'],
                'time series feats': [],
                'drop cols': ['Adj Close']
                }
    
    elif strat == 'simple bollinger band':
        return {'mean type': 'exp',
                'std type': 'rolling',
                'mean price': 'Close',
                'std price': 'Low',
                'mean days': 15,
                'std days': 14,
                'factor': -1.94,
                'profit': 100,
                'stop': -7,
                'max hold': 5,
                
                'return feats': ['Open', 'Low', 'High', 'Close', 'boll_lower',
                                 'Volume'],
                'time series feats': [],
                'drop cols': ['Adj Close']
                }
    
    elif strat == 'bollinger squeeze':
        return {'lower mean type': 'exp',
                'upper mean type': 'rolling',
                
                'lower std type': 'rolling',
                'upper std type': 'exp',
                
                'lower mean price': 'Close',
                'upper mean price': 'Close',
                
                'lower std price': 'Open',
                'upper std price': 'Low',
                
                'lower mean days': 285,
                'upper mean days': 261,
                
                'lower std days': 3,
                'upper std days': 59,

                'lower factor': -3.07,
                'upper factor': 1.06,
                
                'profit': 100,
                'stop': -7,
                'max hold': 5,
                'thresh': 11.65,
                
                'return feats': ['Open', 'Low', 'High', 'Close',
                                 'boll_lower', 'boll_upper', 'Volume'],
                'time series feats': ['boll_diff'],
                'drop cols': ['Adj Close']
                }
    
    '''
    Additional strats
    
    Simple boll band
    
    mean type: exp
    std type: rolling
    mean price: Close
    std price: Low
    mean days: 15
    std days: 14
    factor: -1.9436595465846858
    profit: 10
    stop: -7
    max hold: 5
    
    Out of sample testing results: 
    Average win rate:  52.525820709644485
    Lowest win rate:  0.0
    Average profit:  0.3641878028090984
    Maximum hold time:  5
    Mean hold time:  4.22526880817111
    Mean number of trades:  263.775
    
    10 day strat
    
    'mean type': 'exp',
            'std type': 'exp',
            'mean price': 'Open',
            'std price': 'Open',
            'mean days': 266,
            'std days': 246,
            'factor': -1.09,
            'profit': 100,
            'stop': -7,
            'max hold': 10,
    
    -------------------
    
    MA crossover
    
    Most optimal parameters are: 
    slow type: rolling
    fast type: rolling
    slow price: Close
    fast price: Open
    slow days: 34
    fast days: 31
    profit: 12.317202817404546
    stop: -7
    max hold: 3
    
    
    Most optimal parameters are: 
    slow type: rolling
    fast type: exp
    slow price: Low
    fast price: Close
    slow days: 18
    fast days: 5
    profit: 10
    stop: -7
    max hold: 10
    
    
    '''