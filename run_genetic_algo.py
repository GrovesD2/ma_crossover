from ga import funcs

if __name__ == "__main__":
    
    ga_config = {
                 # Strategy to optimise
                 # Can use:
                 # - simple bollinger band
                 # - simple ma crossover
                 'strat': 'simple bollinger band',
        
                 # Basic controls for the genetic algo
                 'num strats': 100, # Number of strategies to try on each evolution
                 'num tickers': 10, # Number of tickers to optimise over
                 'num evolutions': 50, # Number of evolutions to perform
                 'keep perc': 0.2, # Percentage of top models to keep on each evolution
                 
                 # What to optimise, can be 'win rate', 'avg profit', 'median profit'
                 'fitness': 'avg profit',
                 
                 # Constraints
                 'max hold': 10, # Maximum number of holding days
                 'min trades': 10, # Minimum trades the strategy performs per ticker
                 'max stop': -10, # Maximum stop loss to consider per trade
                 'min profit': 2, # Minimum profit target per trade
                 
                 # Out of sample testing and saving name
                 'num tickers test': 200, # Number of tickers to perform the out of sample testing
                 'save name': 'profit_boll_short', # Save name for the optimised params
                 }

    # Run the algorithm
    strat = funcs.main(ga_config)

    # Perform the out of sample test
    funcs.out_sample_test(strat, ga_config)