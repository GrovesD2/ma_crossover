# ma_crossover

A simple repository of codes to tune a moving average crossover technique for stock-market trading (no $1,000,000 groundbreaking methods sadly!).

**For now, this is all WIP!**

This repo includes:
- A dashboard to run a moving average crossover strategy (given a ticker), view the strategy stats, and plot the candlestick charts for the trades made.
- A genetic algorithm to tune the hyperparameters on a set of tickers.
- A neural network to predict when a given crossover strategy will work/fail.

## Before using any of the tools
Download ticker data using the `get_ticker_data`, which grants the user the options:
1. Download the relevent requirements from the "requriements.txt" file (**TO DO**) 
2. To download the s&p500 ticker data
3. Download all tickers in the "tickers.csv" file. More can be added to this file, however three are included as standard. **Please make sure there is a single column with the header `ticker` in this csv file.**

## Using the dashboard
I run the dashboard from the anaconda prompt (cd'd into the directory of the repo) using the command `streamlit run strategy_dash.py`. The dashboard requries the ticker data to be available for the ticker selected.

## Using the genetic algorithm
**TO DO**

## Using the neural network
**IN PROGRESS**

## Some notes
1. The buying/selling algorithm has been compiled with `numba`. There is likely an efficient pandas approach to doing this, however, once the numba code has been compiled it is remarkably efficient.