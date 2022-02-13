# ma_crossover

**For now, this is all WIP!**

A simple repo of code to investigate the tuning of a simple two-moving-average crossover trading technique (no $1,000,000 groundbreaking methods sadly!).

This repo includes:
- A dashboard to run a moving average crossover strategy (given a ticker + params), view the strategy stats, and plot the candlestick charts for the trades made.
- A neural network to try and learn when a given crossover strategy will work/fail (so far, this avenue is not promising)
- A genetic algorithm to tune the hyperparameters on a select of tickers.

## The trading approach
Simply put, if a faster moving average crosses a slower one from underneath, that's the buy signal. You remain in the trade until either the profit target/stop loss is hit, or the trade is held for a maximum allowed days where you exit on the next days open.

## Before using any of the tools
1. Download the relevent requirements from the "requriements.txt" file (**TO DO**) 
2. Download daily ticker price data using the `get_ticker_data`, which grants the user the options:
    - To download the s&p500 ticker data
    - Download all tickers in the "tickers.csv" file. More can be added to this file, however three are included as standard. **Please make sure there is a single column with the header `ticker` in this csv file.**

## The dashboard
The dashboard can be from the anaconda prompt (cd'd into the directory of the repo) using the command `streamlit run strategy_dash.py`. The dashboard requries the ticker data to be available for the ticker selected. Controllable parameters are on the left-hand-side of the dashboard, outputs from the strategy and a candlestick chart for each trade is given on the right-hand-side

## The neural network
The principle idea behind the nn approach is to learn the trading set-ups where the moving average crossover strategy will work. So rather than predicting raw price increases, the aim is to use a NN to notice patterns where a trading strategy works. **Currently, this is not proving a viable route**, better feature engineering may be required, or, the problem may not be tractable at all!

If someone smarter than me (not hard) improves this code, please let me know!

### To use
1. Configure and run `get_nn_input.py`, which generates a .csv file containing the input training data to the nn.
2. Configure and run `train_nn.py`. This will train and save the nn model. 
**NOTE**: Class label 0 = stop loss hit, class label 1 = max holding days passed, class label 2 = profit target hit.

## Using the genetic algorithm
**TO DO**

## Some notes
1. The buying/selling algorithm has been compiled with `numba`. There is likely an efficient pandas approach to doing this, however, once the numba code has been compiled it is remarkably efficient.
2. This approach may be extendable (and perform better) with other trading strategies, this is future work and hasn't been attempted yet.
