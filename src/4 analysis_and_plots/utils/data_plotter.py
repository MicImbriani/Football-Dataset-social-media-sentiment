import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.api.indexers import BaseIndexer
import numpy as np
import re

# returns a dataframe where column names are space separated
def replace_with_space_separated_columns(df):
    columns_dict = {}
    for c in df.columns:
        if "_" in c:
            res = c.replace("_", " ")
        else:
            res = re.sub('([A-Z])', r' \1', c)
        columns_dict[c] =  res[:1].upper() + res[1:].lower()

    return df.rename(columns=columns_dict)

# kde estimate of df
def plot_kde(df, ax, color = None):
    if color is not None:
        return df.plot.kde(ax=ax, color=color,alpha=0.5)
    else:
        return df.plot.kde(ax=ax,alpha=0.5)

def plot_hist(df, ax, color = None):
    if color is not None:
        return df.plot.hist(ax=ax, color=color,alpha=0.5,stacked=True)
    else:
        return df.plot.hist(ax=ax,alpha=0.5,stacked=True)


# Basic shapes and description of a dataframe
def print_shapes_and_basic_stats(df):
    print(f"shape:\n {df.shape}\n")
    print(f'Description:\n {df.describe()}\n')

# Correlations in a heatmap of a dataframe
def plot_correlations(df, ax):
    df = replace_with_space_separated_columns(df)
    corr_matrix = df.corr()
    print(df.columns)
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))

    return sns.heatmap(corr_matrix, 
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns, cmap="BrBG", mask = mask,
        vmin=-1, vmax=1, ax=ax,annot=True,fmt=".1f")

def plot_correlations_distinct(df, ax):
    df = replace_with_space_separated_columns(df)
    corr_matrix = df.corr()
    columns = corr_matrix.columns

    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
    mask = mask[13:-1, 1:-9]
    corr_matrix = corr_matrix.iloc[13:-1, 1:-9]
    print(corr_matrix)
    print("mask",mask.shape)

    print("matrix",corr_matrix.shape)
    print(columns[0:13])
    print("-––--------––--------––-------")
    print(columns[13:21])

    return sns.heatmap(corr_matrix,
         cmap="BrBG",
        vmin=-1, vmax=1, ax=ax,annot=True,fmt=".1f")

    
# Date plotting-formatter for the x-axis of a 2D plot
def format_dates():
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=90)

# Plotting function that takes a list of y_axes columns to plot against a single x_axis
# minmax_scale is if all columns in y_axis should be scaled between [0,1]
# line is whether or not to use a lineplot of scatterplot
# If the x_axis is a date it is formatted accordingly
def plot_cols(df, x_axis: str, y_axes: list[str], minmax_scale = False, line=False):
    for y in y_axes:
        data_y  = MinMaxScaler().fit_transform(df[[y]]) if minmax_scale else df[y]
        if line:
            plt.plot(df[x_axis], data_y, label=y)
        else:
            plt.scatter(df[x_axis], data_y, label=y, s=[0.3 for i in range(len(df))])
    if df.dtypes[x_axis] == 'datetime64[ns, UTC]':
        format_dates()
    plt.xlabel(x_axis)
    plt.legend()

# Autocorrelation plot of a specific column in a dataframe
def plot_auto_correlation(df, col: str):
    pd.plotting.autocorrelation_plot(df.loc[:,[col]])

# Plots the decomposed trends and seasonality of a col in a dataframe based on the period
def plot_seasonality(df, col: str, period: int):
    decomposition_result = seasonal_decompose(df[col], model='additive', period=period)
    observed = decomposition_result.observed
    seasonal = decomposition_result.seasonal
    trend = decomposition_result.trend
    resid = decomposition_result.resid

    fig,axs = plt.subplots(6,1,figsize=(8,6))
    axs[0].plot(df['date'],observed,'-',linewidth=1,label='raw',color='#9FC0DE')
    axs[0].set_ylabel('Signal')
    axs[1].plot(df['date'],trend,'-',linewidth=1,label='raw',color='#FF985A')
    axs[1].set_ylabel('Trend')
    axs[2].plot(df['date'],observed-trend,'-',linewidth=1,label='raw',color='#FFC3C3')
    axs[2].set_ylabel('Detrended')
    axs[3].plot(df['date'],seasonal,'-',linewidth=1,label='raw',color='#89E3CC')
    axs[3].set_ylabel('Seasonality')
    axs[4].plot(df['date'],resid,'-',linewidth=1,alpha=0.8,label='raw',color='#C580BB')
    axs[4].set_ylabel('Residuals')
    axs[5].plot(df['date'],trend+seasonal,'-',linewidth=1,alpha=0.5,label='raw',color='black')
    axs[5].set_ylabel('Trend + \nSeason')         
    plt.tight_layout()


# Plots the rolling means according to the mean value of tweets inbetween the previous and next game
def plot_rolling_means_of_tweets_between_games(tweets, games, y_axes: list[str], minmax_scale=False, line=True):
    # An Indexer used for rolling operations on dataframes
    # For each row in a tweets dataframe, 
    # a start and end index is returned according to the nearest future and previous games
    class GameIndexer(BaseIndexer):
        def __init__(self, tweets, games):
            super().__init__()
            self.tweets = tweets
            self.games = games

        def get_window_bounds(self, num_values, min_periods, center, closed):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                previous_games_indices = self.games.index[self.games['date'] <= self.tweets.loc[i]['date']]
                next_games_indices = self.games.index[self.games['date'] > self.tweets.loc[i]['date']]
                if len(previous_games_indices) == 0:
                    start[i] = 0
                else:
                    index_start = self.tweets.index[self.tweets['date'] == self.games.loc[previous_games_indices[-1]]['date']]
                    start[i] = index_start[0]
                if len(next_games_indices) == 0:
                    end[i] = len(end)
                else:
                    index_end = self.tweets.index[self.tweets['date'] == self.games.loc[next_games_indices[0]]['date']]
                    end[i] = index_end[0] - 1
            return start, end


    indexer = GameIndexer(tweets, games)
    mean_by_games = tweets.rolling(indexer).mean()
    mean_by_games['date'] = tweets['date']
    plot_cols(mean_by_games, 'date', y_axes, minmax_scale, line)
