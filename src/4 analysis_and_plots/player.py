# This file contains logic for data management and plotting for (all) players
# Users can instantiate a Player, create a figure with axes, and use the plotting functions
# to plot basic things on the axes. The axes are returned so they can be styled later

from utils import data_plotter
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats.mstats as mstats

data_path = "../../data/final"


# Enum for managing dataframes within the Player class
class DataEnum(Enum):
    TWEETS_AGG = "aggregated tweets"
    TWEETS_RAW = "raw tweets"
    STATS = "game stats"


# Wrapper class for handling all players individually and combined
class AllPlayers:
    def __init__(self):
        # Individual players
        self.Cristiano_Ronaldo = Player('Cristiano Ronaldo')
        self.Erling_Haaland = Player('Erling Haaland')
        self.Kevin_De_Bruyne = Player("Kevin De Bruyne")
        self.Kylian_Mbappé = Player('Kylian Mbappé')
        self.Lionel_Messi = Player('Lionel Messi')
        self.Neymar = Player('Neymar')
        self.Riyad_Mahrez =Player('Riyad Mahrez')
        self.Robert_Lewandowski = Player('Robert Lewandowski')
        self.Sadio_Mané = Player('Sadio Mané')
        self.Virgil_van_Dijk = Player('Virgil van Dijk')

        # All players combined
        self.combined = Player("combined", read_values=False)
        self.all = [self.Cristiano_Ronaldo, self.Erling_Haaland, self.Kevin_De_Bruyne,
                    self.Kylian_Mbappé, self.Lionel_Messi, self.Neymar, self.Sadio_Mané,
                    self.Virgil_van_Dijk,
                    self.Robert_Lewandowski]
        self.combined.combine_with(self.all)




def read_player_tweets_all(player: str): 
    path = f"../../../data/final/tweets/all/{player}.csv"
    df = pd.read_csv(path, parse_dates=['date'], lineterminator='\n')
    df.set_index('date')
    return df

def read_player_tweets_aggregated(player: str):
    path = f"../../../data/final/tweets/aggregated/{player}.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date')
    return df

def read_player_games(player: str):
    path = f"../../../data/final/games/{player}.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date')
    return df


# Class for managing and plotting player data
# Utilizes builder-like features, such that you can combine player data
class Player:
    def __init__(self, name, read_values=True):
        self.name = name
        if read_values:
            tweets_agg_path = os.path.join(data_path, "tweets", "aggregated", f"{name}.csv")
            self.aggregated_tweets = pd.read_csv(tweets_agg_path, parse_dates=['date']) 
            self.aggregated_tweets.set_index('date')

            tweets_all_path = os.path.join(data_path, "tweets", "all", f"{name}.csv")
            self.raw_tweets = pd.read_csv(tweets_all_path, parse_dates=['date'])
            self.raw_tweets.set_index('date')

            games_path = os.path.join(data_path, "games", f"{name}.csv")
            self.player_stats = pd.read_csv(games_path, parse_dates=['date'])
            self.player_stats.set_index('date')
        else:
            self.aggregated_tweets = self.raw_tweets = self.player_stats = None


    # ============================= DATA MANAGEMENT =============================#
    def get_data(self, data_type: DataEnum):
        match data_type:
            case DataEnum.TWEETS_AGG:
                return self.aggregated_tweets
            case DataEnum.TWEETS_RAW:
                return self.raw_tweets
            case DataEnum.STATS:
                return self.player_stats

    # Returns a dataframe of aggregated tweets and player stats merged on game days and player name
    # Lag can be used to specify if the matching days are shifted
    def __get_aggregated_tweets_on_game_days(self, lag=0):
        temp_tweets_aggregated = self.aggregated_tweets.copy(deep=True)
        temp_tweets_aggregated['date'] = temp_tweets_aggregated['date'].shift(lag)
        games_with_tweets = self.player_stats.merge(temp_tweets_aggregated, 'left',
                                                    left_on=["date", 'name'],
                                                    right_on=['date', 'player_name'])
        return games_with_tweets

    # Applies a rolling on a dataframe that contains matches and tweets
    # window is the window_size, e.g. how many matches to apply the correlation over
    # min_period is how many matches we want as minimum, before we "cut" the ends of the timeseries
    # corr_key is the column you want to compute all the correlations for
    # It is assumed that window and min_period is uneven and that the window is bigger
    def __get_rolling_correlation_of_tweets_on_game_days(self, window: int,
                                                         min_periods: int,
                                                         corr_key='rating'):
        assert window >= min_periods and window % 2 == 1 and min_periods % 2 == 1
        games_with_tweets = self.__get_aggregated_tweets_on_game_days(lag=0)
        rolling_mean = games_with_tweets.rolling(window=window, min_periods=min_periods,
                                                 center=False).mean()
        out = pd.DataFrame(columns=np.append(games_with_tweets.corr().columns.values,
                                             ['date', 'avg_positive_on_day',
                                              'avg_positive_rolling_mean',
                                              'rating_on_day', 'rating_rolling_mean']))
        sides = window // 2
        min_sides = min_periods // 2
        for i in range(len(games_with_tweets)):
            if i - min_sides < 0 or i + min_sides > len(games_with_tweets) - 1:
                continue
            data = games_with_tweets.loc[i - sides:i + sides]
            out_i = 0 if pd.isnull(out.index.max()) else out.index.max() + 1
            out.loc[out_i] = np.append(data.corr()[corr_key].values,
                                       [games_with_tweets.loc[i]['date'],
                                        games_with_tweets.loc[i]['avg_positive'],
                                        rolling_mean.loc[i]['avg_positive'],
                                        games_with_tweets.loc[i]['rating'],
                                        rolling_mean.loc[i]['rating']
                                        ])
        return out

    # Combines a datatype with a list of other players
    def __combine_datatype(self, data_type: DataEnum, players: list['Player']):
        others = [p.get_data(data_type) for p in players]
        others.insert(0, self.get_data(data_type))
        new = pd.concat(others, axis=0)
        return new.sort_values('date')

    # Combines all datatypes with a list of other players
    def combine_with(self, players: list['Player']):
        self.aggregated_tweets = self.__combine_datatype(DataEnum.TWEETS_AGG, players)
        self.raw_tweets = self.__combine_datatype(DataEnum.TWEETS_RAW, players)
        self.player_stats = self.__combine_datatype(DataEnum.STATS, players)

        

    # ============================= PLOTTING FUNCTIONS =============================#
    # Plots a kde estimate of a specific column in the dataframe given by data_type.
    # Trim can be used to trim upper/lower percentile of data before plotting
    def plot_col_kde(self, data_type: DataEnum, col: str, ax = None, trim = None, color = None):
        data = self.get_data(data_type)[col].copy()
        if trim is not None:
            low_perc = np.percentile(data, trim[0])
            high_perc = np.percentile(data, trim[1])
            data = data[(data >= low_perc) & (data <= high_perc)]
        return data_plotter.plot_kde(data, ax=ax, color=color)

    def plot_col_hist(self, data_type: DataEnum, col: str, ax = None, trim = None,
                   color = None):
        data = self.get_data(data_type)[col].copy()
        if trim is not None:
            low_perc = np.percentile(data, trim[0])
            high_perc = np.percentile(data, trim[1])
            data = data[(data >= low_perc) & (data <= high_perc)]
        return data_plotter.plot_hist(data, ax=ax, color=color)

    # Returns basic data charactestics given a data_type
    def get_shape_and_basic_stats(self, data_type: DataEnum):
        shape = self.get_data(data_type).shape
        basic_stats = self.get_data(data_type).describe()
        return shape, basic_stats

    # Plots correlations in the dataframe given by data_type
    def plot_correlations(self, data_type: DataEnum, ax=None):
        return data_plotter.plot_correlations(self.get_data(data_type), ax=ax)

    # Plots correlations between game stats and tweets on the same days
    def plot_games_with_tweets_correlations(self, lag=0, ax=None):
        return data_plotter.plot_correlations_distinct(self.__get_aggregated_tweets_on_game_days(
            lag), ax=ax)

    # Plots correlations betwen game stats and tweets 6 days before - 6 days after the games
    def plot_games_with_tweets_correlation_varying_lag(self, tweet_x_axis: str, ax=None, color=None, columns = None):
        if columns == None:
            columns=['minutes', 'rating', 'shotsTotal', 'shotsOn', 'goals', 'assists',
                     'passes',
                     'tackles', 'blocks', 'duelsTotal', 'duelsWon', 'foulsDrawn',
                     'penaltiesScored']
        lag_df = pd.DataFrame(columns=columns)            
        for l in range(-6, 7, 1):
            df = self.__get_aggregated_tweets_on_game_days(lag=l)
            lag_df.loc[l] = df.corr()[tweet_x_axis][columns]

        lag_df = data_plotter.replace_with_space_separated_columns(lag_df)
        ax = lag_df.plot(ax=ax, color=color)
        ax.legend(loc='upper left')
        return ax


    # ============================= TODO / DISCUSS PLOTTING FUNCTIONS =============================#

    # def plot_cols_of_data(self, data_type: DataEnum, x_axis: str, y_axes: list[str],
    #                       minmax_scale=False, line=False):
    #     data_plotter.plot_cols(self.get_data(data_type), x_axis, y_axes, minmax_scale,
    #                            line)
    #     plt.title(f'{self.name}: {data_type.value}')
    #     plt.show()

    # def plot_auto_correlation(self, data_type: DataEnum, col: str):
    #     data_plotter.plot_auto_correlation(self.get_data(data_type), col)
    #     plt.title(f'{self.name}: Autocorrelation of {col} in {data_type.value}')

    #     plt.show()

    # def plot_games_with_tweets(self, x_axis: str, y_axes: list[str], minmax_scale=False,
    #                            line=False, lag=0):
    #     games_with_tweets = self.__get_aggregated_tweets_on_game_days(lag)
    #     data_plotter.plot_cols(games_with_tweets, x_axis, y_axes, minmax_scale, line)
    #     plt.title(f'{self.name}: Lag {lag} aggregated tweets on game days')


    #     plt.show()

    # def plot_seasonality(self, data_type: DataEnum, col: str, period: int, savefigure=False):
    #     data_plotter.plot_seasonality(self.get_data(data_type), col, period)
    #     plt.title(f'{self.name}: Decomposed seasonality of {col} in {data_type.value}')
    #     if savefigure:
    #         plt.savefig(
    #             os.path.join("plots/seasonality", self.name.replace(" ", "_")))
    #     plt.show()

    # def plot_rolling_means_for_tweets_between_matches(self, y_axes: list[str],
    #                                                   minmax_scale=False, line=True):
    #     data_plotter.plot_rolling_means_of_tweets_between_games(self.aggregated_tweets,
    #                                                             self.player_stats,
    #                                                             y_axes, minmax_scale,
    #                                                             line)
    #     plt.title(f'{self.name}: Rolling means for tweets inbetween games')
    #     plt.show()

    # def plot_rolling_means_given_window(self, y_axes: list[str], window: int,
    #                                     minmax_scale=False, line=True):
    #     mean_df = self.aggregated_tweets.rolling(window=window, min_periods=1,
    #                                              win_type='boxcar', center=True).mean()
    #     mean_df['date'] = self.aggregated_tweets['date']
    #     data_plotter.plot_cols(mean_df, 'date', y_axes, minmax_scale, line)
    #     plt.title(f'{self.name}: Rolling means for tweets. Window = {window}')

    # # Plots the rolling correlation with vertical lines at the dates of events.
    # # events is assumed to be a dict(datetime, string)
    # # keys are the time of events and values are event descriptions
    # def plot_rolling_correlation_with_events(self, events, window: int,
    #                                          min_periods: int,
    #                                          corr_key=['avg_positive'],
    #                                          minmax_scale=False, line=True):
    #     rolling_corr = self.__get_rolling_correlation_of_tweets_on_game_days(window,
    #                                                                          min_periods)
    #     data_plotter.plot_cols(rolling_corr, 'date', corr_key, minmax_scale, line)
    #     for k, v in events.items():
    #         plt.axvline(k, c='red')
    #     # plt.title(f'{self.name}: Rolling means for tweets. Window = {window}')
    #     plt.show()
