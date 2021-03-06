from ga import funcs

if __name__ == "__main__":
    
    ga_config = {
                 # Strategy to optimise
                 # Can use:
                 # - bollinger squeeze
                 # - simple bollinger band
                 # - simple ma crossover
                 'strat': 'simple ma crossover',
        
                 # Basic controls for the genetic algo
                 'num strats': 50, # Number of strategies to try on each evolution
                 'num tickers': 15, # Number of tickers to optimise over
                 'num evolutions': 50, # Number of evolutions to perform
                 'keep perc': 0.2, # Percentage of top models to keep on each evolution
                 
                 # What to optimise, can be 'win rate', 'avg profit', 'median profit'
                 'fitness': 'win rate',
                 
                 # Constraints
                 'max hold': 10, # Maximum number of holding days
                 'min trades': 40, # Minimum trades the strategy performs per ticker
                 'max stop': -7, # Maximum stop loss to consider per trade
                 'min profit': 10, # Minimum profit target per trade
                 
                 # Out of sample testing and saving name
                 'num tickers test': 200, # Number of tickers to perform the out of sample testing
                 'save name': 'ma_win_rate', # Save name for the optimised params
                 }

    # Run the algorithm
    strat = funcs.main(ga_config)

    # Perform the out of sample test
    funcs.out_sample_test(strat, ga_config)