import pandas as pd
import streamlit as st

from utils import dash, strategy

# Initialise the dictionary for the config opts
config = {}

# Sidebar --------------------------------------------------------------------
st.sidebar.title('Strategy Opts')

ticker = st.sidebar.text_input('Input ticker name here:',
                               value = 'TSLA',
                               )

# Slow moving average options
st.sidebar.write('Slow MA Opts')
config['slow type'] = st.sidebar.selectbox('Slow MA type',
                                           ('rolling', 'exp'),
                                           )
config['slow price'] = st.sidebar.selectbox('Slow Price field',
                                           ('Open', 'Low', 'High', 'Close'),
                                           )
config['slow days'] = st.sidebar.number_input('Slow MA days',
                                               value = 50,
                                               min_value = 2,
                                               max_value = 300,
                                               step = 1,
                                               )
# Fast moving average options
st.sidebar.write('Fast MA Opts')
config['fast type'] = st.sidebar.selectbox('Fast MA type',
                                           ('rolling', 'exp'),
                                           )
config['fast price'] = st.sidebar.selectbox('Fast Price field',
                                           ('Open', 'Low', 'High', 'Close'),
                                           )
config['fast days'] = st.sidebar.number_input('Fast MA days',
                                              value = 10,
                                              min_value = 2,
                                              max_value = 300,
                                              step = 1,
                                              )

# Other strategy opts
st.sidebar.write('Strategy Opts')
config['profit'] = st.sidebar.number_input('Profit target (%)',
                                           value = 10.,
                                           min_value = 1.,
                                           max_value = 300.,
                                           step = 0.1,
                                           )
config['stop'] = - st.sidebar.number_input('Stop loss (%)',
                                           value = 5.,
                                           min_value = 0.,
                                           max_value = 100.,
                                           step = 0.1,
                                           )
config['max hold'] = st.sidebar.number_input('Maximum holding days',
                                             value = 20,
                                             min_value = 1,
                                             max_value = 100,
                                             step = 1,
                                             )

# The moving average days require changing to integers, so that they can be
# passed through the code
config['slow days'] = int(config['slow days'])
config['fast days'] = int(config['fast days'])

# Run the strategy -----------------------------------------------------------

df = pd.read_csv(f'data/{ticker}.csv')
df = strategy.add_strat_cols(df, config)
df_summary, stats = strategy.run_strategy(df, config)

# Main dash ------------------------------------------------------------------

st.subheader('Strategy Statistics')
st.write(stats)

# This number input allows us to select which candlestick chart to plot, based
# on the trades printed from df_summary (i.e. each number corresponds to the
# index of the dataframe)
plot_case = st.number_input('Select which trade to plot (see dataframe below)',
                            value = 0,
                            min_value = 0,
                            max_value = int(stats['number of trades']),
                            step = 1,
                            )

st.subheader(f'Candlestick Chart for Trade {int(plot_case)}')

# Filter the dataframe based on the buy date (pm 1 month), and get the buy and
# sell info for the candlestick chart
df_filt, bought, sold = dash.get_plotting_info(df,
                                               df_summary,
                                               int(plot_case))

# Obtain the candlestick plot
fig2 = dash.get_candlestick_plot(df_filt,
                                 ticker,
                                 bought,
                                 sold)
st.plotly_chart(fig2, use_container_width=True)

st.subheader('Each Trade')
st.write(df_summary)

