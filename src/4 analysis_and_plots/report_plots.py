# This file reproduces all plots used in the rapport, and prints basic data statistics
import argparse
import os
from player import Player, AllPlayers, DataEnum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator
from utils.data_plotter import replace_with_space_separated_columns
from matplotlib.patches import Rectangle

# TODO make plots
# TODO clean up all files in analysis folder
# TODO decide what to do with the twitter_exploration file
# TODO remove text column for all tweets
# TODO add guide on how to recollect tweets


# Structure:
# - Show ranges over data in rapport in a tabel (both tweets and player stats)
# 	- Refer to appendix for distrubtions over all player stats combined for all players
# 	- Refer to appendix for distrubtions over all player stats combined for all players
# - Explain that the data distribution is different for player position
# 	- Show distribution plot for specific collumns of the games stats for 3 different players
# 		- Example: Distrubtions for goals for 3 players in same plot
# 		- Example: Distrubtions for tackles for 3 players in same plot
# - Move on to correlation in the data:
# 	- Plot correlation for all tweets for all players combined
#   - Plot correlation for player stats with tweets on game days for specific players (mid, fw, df)


# ===================== BASIC DATA STATISTICS =====================
def basic_data_statistics(players: AllPlayers, save_dir):
    agg_tweets_shape, agg_tweets_stats = players.combined.get_shape_and_basic_stats(DataEnum.TWEETS_AGG)
    raw_tweets_shape, raw_tweets_stats = players.combined.get_shape_and_basic_stats(DataEnum.TWEETS_RAW)
    games_shape, games_stats = players.combined.get_shape_and_basic_stats(DataEnum.STATS)
    pd.set_option('display.max_columns', None)
    with open(os.path.join(save_dir, "combined_data_stats.txt"), "w") as f:
        f.write("Aggregated Tweets:\n")
        f.write(f'======================================\n')
        f.write(f"Shape: {agg_tweets_shape}\n")
        f.write(f'Stats / description:\n')
        f.write(f'{agg_tweets_stats}\n\n\n')

        f.write("All Tweets:\n")
        f.write(f'======================================\n')
        f.write(f"Shape: {raw_tweets_shape}\n")
        f.write(f'Stats / description:\n')
        f.write(f'{raw_tweets_stats}\n\n\n')  

        f.write("Games:\n")
        f.write(f'======================================\n')
        f.write(f"Shape: {games_shape}\n")
        f.write(f'Stats / description:\n')
        f.write(f'{games_stats}')      
        f.close()
    pd.reset_option('display.max_columns')


def number_of_tweets_for_each_player(players: AllPlayers, save_dir):
    with open(os.path.join(save_dir, "count_player_tweets.txt"), "w") as f:
        for player in players.all:
            f.write(f"# Tweets for {player.name} = {player.raw_tweets.shape[0]}\n")     
        f.close()

def number_of_total_sentiments_for_each_player(players: AllPlayers, save_dir):
    with open(os.path.join(save_dir, "count_player_total_sentiments.txt"), "w") as f:
        for player in players.all:
            f.write(f"# negative for {player.name} = {player.aggregated_tweets.total_negative.sum()}\n")
            f.write(f"# neutral for {player.name} = {player.aggregated_tweets.total_neutral.sum()}\n")    
            f.write(f"# positive for {player.name} = {player.aggregated_tweets.total_positive.sum()}\n\n")         
        f.close()



# ===================== DATA SHAPES / PROBABILITY DENSITIES / HISTOGRAMS =====================
def combined_probability_densities(players: AllPlayers, save_dir):
    # Game stats
    fig, axes = plt.subplots(5, 4, figsize=(2,2))
    wanted_columns = players.combined.player_stats.columns[3:] # Ignore names and dates

    for i, ax in enumerate(axes.flatten()):
        ax = players.combined.plot_col_kde(DataEnum.STATS, wanted_columns[i], ax, trim=(0, 100))
        ax.set_xlabel(wanted_columns[i])


    fig.suptitle('Probability densities for game stats combined for all players')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_game_stats_densities.png"))

    # Aggregated tweets
    fig, axes = plt.subplots(3, 3, figsize=(7,7))
    wanted_columns = players.combined.aggregated_tweets.columns[2:]
    for i, ax in enumerate(axes.flatten()):
        ax = players.combined.plot_col_kde(DataEnum.TWEETS_AGG, wanted_columns[i], ax, trim=(0, 99))
        ax.set_xlabel(wanted_columns[i])

    fig.suptitle('Probability densities for aggregated tweets combined for all players')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_aggregated_twitter_densities.png"))

def combined_probability_densities_1(players: AllPlayers, save_dir):
    # Game stats
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    wanted_columns = players.combined.player_stats.columns[3:]  # Ignore names and dates

    # for i, ax in enumerate(axes.flatten()):
    #     ax = players.combined.plot_col_kde(DataEnum.STATS, wanted_columns[i], ax, trim=(0, 100))
    #     ax.set_xlabel(wanted_columns[i])

    axes[0, 0] = players.combined.plot_col_hist(DataEnum.STATS, [
        'minutes'], axes[0, 0], trim=(0, 100))

    axes[0, 1] = players.combined.plot_col_hist(DataEnum.STATS, ["rating"], axes[0, 1],
                                                trim=(0, 100))
    axes[1, 0] = players.combined.plot_col_hist(DataEnum.STATS, ['shotsTotal',
                                                                 'shotsOn', 'goals',
                                                                 'assists'], axes[1, 0],
                                                trim=(0, 100))
    axes[1, 0].semilogy()
    axes[1, 1] = players.combined.plot_col_hist(DataEnum.STATS, ['passes'], axes[1, 1],
                                                trim=(0, 100))
    axes[2, 0] = players.combined.plot_col_hist(DataEnum.STATS, ['tackles', 'blocks'],
                                                axes[2, 0], trim=(0, 100))
    axes[2, 1] = players.combined.plot_col_hist(DataEnum.STATS, ['duelsTotal',
                                                                 'duelsWon',
                                                                 'foulsDrawn'], axes[2,
                                                                                     1],
                                                trim=(0, 100))
    fig.suptitle('Probability densities for game stats combined for all players')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_combined_game_stats_densities.png"))

    # Aggregated tweets
    fig, axes = plt.subplots(2, 2, figsize=(12,8))

    axes[0, 0] = players.combined.plot_col_hist(DataEnum.TWEETS_AGG, [
        'avg_positive','avg_neutral','avg_negative'], axes[0, 0], trim=(0, 100))

    axes[0, 1] = players.combined.plot_col_hist(DataEnum.TWEETS_AGG, ['avg_likes'],
                                                axes[0,1],
                                                trim=(0, 100))
    axes[0, 1].semilogy()

    axes[1, 0] = players.combined.plot_col_hist(DataEnum.TWEETS_AGG, ['total_tweets',
                                                                      'total_negative','total_neutral','total_positive'],
                                                axes[1, 0],
                                                trim=(0, 100))
    axes[1, 1] = players.combined.plot_col_hist(DataEnum.TWEETS_AGG, ['count'], axes[1, 1],
                                                trim=(0, 100))
    axes[1, 1].semilogy()

    fig.suptitle('Probability densities for aggregated tweets combined for all players')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_"
                                       "combined_aggregated_twitter_densities.png"))

def combined_probability_densities_tweets(players: AllPlayers, save_dir):
    fig,ax2 = plt.subplots(1, 1, figsize=(8,4))
    stat_1_titel = "tweets"
    stat_2_titel = "total"

    # ax1 =  players.combined.plot_col_kde(DataEnum.TWEETS_AGG, "avg_positive", ax1,
    #                                      trim=(0, 99))
    # players.combined.plot_col_kde(DataEnum.TWEETS_AGG, "avg_negative", ax1,
    #                               trim=(0, 99))
    # players.combined.plot_col_kde(DataEnum.TWEETS_AGG, "avg_neutral", ax1,
    #                               trim=(0, 99))
    #
    # ax1.set_title(f"Distribution for {stat_1_titel}")
    # ax1.set_ylabel("Probability Density")
    # ax1.set_xlabel("Tweet sentiment")
    # plt.legend(['Avg Positive', 'Avg Negative', 'Avg Neutral'])

    # ax2 = players.combined.plot_col_kde(DataEnum.TWEETS_AGG, "total_tweets", ax2,
    #                                     trim=(0, 99))
    # players.combined.plot_col_kde(DataEnum.TWEETS_AGG, "total_positive", ax2,
    #                               trim=(0, 99))
    # players.combined.plot_col_kde(DataEnum.TWEETS_AGG, "total_negative", ax2,
    #                               trim=(0, 99))
    # players.combined.plot_col_kde(DataEnum.TWEETS_AGG, "total_neutral", ax2,
    #                               trim=(0, 99))
    #
    # ax2.set_title(f"Distribution for aggregated tweets")
    # ax2.set_ylabel("Probability Density")
    # ax2.set_xlabel("Total tweets")
    # plt.legend(['Total Tweets', 'Total Positive', 'Total Negative', 'Total Neutral'])
    #
    # plt.savefig(os.path.join(save_dir, "distribution_tweets_total.png"))

    ax2 = players.combined.plot_col_hist(DataEnum.TWEETS_AGG, "count", ax2,
                                        trim=(0, 99))
    players.combined.plot_col_hist(DataEnum.TWEETS_AGG, "avg_likes", ax2,
                                  trim=(0, 99))


    ax2.set_title(f"Distribution for aggregated tweets")
    ax2.set_ylabel("Probability Density")
    ax2.set_xlabel("Total tweets")
    plt.semilogy()
    plt.legend(['Total amount of tweets', 'Average amount of positive'])

    plt.savefig(os.path.join(save_dir, "test.png"))

def combined_probability_densities_stats(players: AllPlayers, save_dir):
    # fig,ax1 = plt.subplots(1, 1, figsize=(8,4))
    #
    # ax1 =  players.combined.plot_col_hist(DataEnum.STATS, ["shotsTotal","shotsOn",
    #                                                       "goals","assists"], ax1,
    #                                      trim=(0, 99))
    #
    #
    # ax1.set_title(f"Distribution for Offensive Stats")
    # ax1.set_ylabel("Amount")
    # ax1.set_xlabel("Count")
    # plt.legend(['Shots Total','Shots On Goal', 'Goals', 'Assists'])

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    ax1 =  players.combined.plot_col_hist(DataEnum.STATS, ["duelsTotal","duelsWon",
                                                           ], ax1,
                                         trim=(0, 99))


    ax1.set_title(f"Distribution for Offensive Stats")
    ax1.set_ylabel("Amount")
    ax1.set_xlabel("Count")
    plt.legend(["Tackles", "Blocks", 'Fouls Drawn'])


    plt.savefig(os.path.join(save_dir, "distribution_stats_def.png"))

def specific_stat_probability_densities(players: AllPlayers, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    stat_1, stat_1_titel = "shotsTotal", "Shots Total"
    stat_2, stat_2_titel = "passes", "Passes"

    ax1 = players.Cristiano_Ronaldo.plot_col_kde(DataEnum.STATS, stat_1, ax1)
    players.Kevin_De_Bruyne.plot_col_kde(DataEnum.STATS, stat_1, ax1)
    players.Virgil_van_Dijk.plot_col_kde(DataEnum.STATS, stat_1, ax1)
    ax1.set_title(f"Distribution for '{stat_1_titel}' for different positions")
    ax1.set_ylabel("Probability Density")
    ax1.set_xlabel(stat_1_titel)


    ax2 = players.Cristiano_Ronaldo.plot_col_kde(DataEnum.STATS, stat_2, ax2)
    players.Kevin_De_Bruyne.plot_col_kde(DataEnum.STATS, stat_2, ax2)
    players.Virgil_van_Dijk.plot_col_kde(DataEnum.STATS, stat_2, ax2)
    ax2.set_title(f"Distribution for '{stat_2_titel}' for different positions")
    ax2.set_ylabel("Probability Density")
    ax2.set_xlabel(stat_2_titel)

    plt.legend(["Cristiano Ronaldo (FW)", "Kevin De Bruyne (MID)", "Virgil van Dijk (DF)"])
    plt.savefig(os.path.join(save_dir, "specific_game_stats_densities.png"))




# ===================== CORRELATIONS =====================
def correlation_simple_3(players : AllPlayers, save_dir):
    # players= {'Christiano Ronaldo': players.Cristiano_Ronaldo, 'Kevin De '
    #                                                            'Bruyne':
    #     players.Kevin_De_Bruyne, 'Virgil van Dijk ': players.Virgil_van_Dijk}

    all = players.combined

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    ax1 = all.plot_games_with_tweets_correlations()
    # ax2 = players.Kevin_De_Bruyne.plot_correlations(DataEnum.STATS, ax2)


    # ax1.set_title("Correlation Stats with Count")
    # ax1.add_patch(Rectangle((0, 2), 1, 3, fill=False, ec="yellow", linewidth=2))

    # ax1.annotate('Total count with avg. likes',
    #              xy=(2, 8), xytext=(4, 4),
    #              fontsize=12, fontweight='light',
    #              arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)

    ax1.set_title("Correlation for Kevin de Bruyne All Tweets and Player Stats")
    # ax2.set_title("Kevin De Bruyne Stats")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"correlation/",
                             "All_Players_Stats_Tweets.jpg"))

def correlation_simple_2(players : AllPlayers, save_dir):
    # players= {'Christiano Ronaldo': players.Cristiano_Ronaldo, 'Kevin De '
    #                                                            'Bruyne':
    #     players.Kevin_De_Bruyne, 'Virgil van Dijk ': players.Virgil_van_Dijk}

    all = players.combined

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    ax1 = all.plot_games_with_tweets_correlations()
    # ax2 = players.Kevin_De_Bruyne.plot_correlations(DataEnum.STATS, ax2)

    # ax1.set_title("Correlation stats with tweets")
    # ax1.add_patch(Rectangle((0, 5), 1, 3, fill=False, ec="blue", linewidth=2))
    # ax1.annotate('Total tweets correlation \nwith total positive, negative and neutral',
    #              xy=(1,5), xytext=(2, 2),
    #              fontsize=12, fontweight='light',
    #              arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)

    # ax1.set_title("Correlation Stats with Count")
    # ax1.add_patch(Rectangle((1, 8), 1, 1, fill=False, ec="yellow", linewidth=2))
    # ax1.annotate('Total count with avg. likes',
    #              xy=(2, 8), xytext=(4, 4),
    #              fontsize=12, fontweight='light',
    #              arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)

    ax1.set_title("Correlation for All Tweets")
    # ax2.set_title("Kevin De Bruyne Stats")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"correlation/",
                             "test_Stats_Tweets.jpg"))

def correlation_simple_1(players : AllPlayers, save_dir):
    # players= {'Christiano Ronaldo': players.Cristiano_Ronaldo, 'Kevin De '
    #                                                            'Bruyne':
    #     players.Kevin_De_Bruyne, 'Virgil van Dijk ': players.Virgil_van_Dijk}

    all = players.combined

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))



    ax1 = all.plot_correlations(DataEnum.TWEETS_AGG, ax1)
    # ax2 = players.Kevin_De_Bruyne.plot_correlations(DataEnum.STATS, ax2)

    ax1.set_title("Correlation stats with tweets")
    ax1.add_patch(Rectangle((0, 5), 1, 3, fill=False, ec="blue", linewidth=2))
    ax1.annotate('Total tweets correlation \nwith total positive, negative and neutral',
                 xy=(1,5), xytext=(2, 2),
                 fontsize=12, fontweight='light',
                 arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)

    ax1.set_title("Correlation Stats with Count")
    ax1.add_patch(Rectangle((1, 8), 1, 1, fill=False, ec="yellow", linewidth=2))
    ax1.annotate('Total count with avg. likes',
                 xy=(2, 8), xytext=(4, 4),
                 fontsize=12, fontweight='light',
                 arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)

    ax1.set_title("Correlation for All Tweets")
    # ax2.set_title("Kevin De Bruyne Stats")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"correlation/",
                             "All_Tweets.jpg"))

def correlation_tweets_stats(players : AllPlayers, save_dir):
    all = players.combined

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))

    ax1 = all.plot_games_with_tweets_correlations(ax= ax1)

    ax1.set_title("Correlation Stats with Tweets")
    ax1.add_patch(Rectangle((13, 18), 1, 3, fill=False, ec = "blue",linewidth=2))
    ax1.annotate('Total tweets correlation \nwith total positive, negative and neutral',
                 xy=(14, 18), xytext=(12,10),
                 fontsize=12, fontweight='light',
                 arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)

    ax1.add_patch(Rectangle((14, 21), 1, 1, fill=False, ec="red", linewidth=2))
    ax1.annotate('Avg. likes - count correlation',
                 xy=(15, 21), xytext=(16, 15),
                 fontsize=12, fontweight='light',
                 arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)

    ax1.add_patch(Rectangle((1, 3), 1, 4, fill=False, ec="purple", linewidth=2))
    ax1.annotate('Rating correlation with offensive stats',
                 xy=(15, 21), xytext=(16, 15),
                 fontsize=12, fontweight='light',
                 arrowprops=dict(arrowstyle='->', lw=1, ls='dashed'), zorder=4)


    plt.tight_layout()
    path = os.path.join(save_dir,"correlation/", "tweet_stats_corr.jpg")
    plt.savefig(path)




# ===================== CONFIDENCE BARS =====================
def game_stats_confidence_bars_3_players(players: AllPlayers, save_dir):
    quantiles = [0.9, 0.95, 0.99]
    fig, axes = plt.subplots(3, 5, figsize=(10,5))
    axes.flatten()[-1].set_visible(False)
    axes.flatten()[-2].set_visible(False)
    plt.subplots_adjust(hspace=1)

    player_colors = {}
    player_colors[players.Cristiano_Ronaldo] = "tomato"
    player_colors[players.Kevin_De_Bruyne] = "darkseagreen"
    player_colors[players.Virgil_van_Dijk] = "cornflowerblue"

    columns = replace_with_space_separated_columns(players.Cristiano_Ronaldo.player_stats).columns[3:]
    for c, ax in zip(columns, axes.flatten()):
        for i, player in enumerate([players.Cristiano_Ronaldo, players.Kevin_De_Bruyne, players.Virgil_van_Dijk]):
            ax.set_ylim([-3, 1])
            ax.spines[['top', 'right', 'left']].set_visible(False)
            stats = replace_with_space_separated_columns(player.player_stats)
            mean = stats[c].mean()
            ax.scatter(mean, -i, marker="o", color=player_colors[player], label=player.name)
            ax.set_yticks([])
            for n, q in enumerate(quantiles):
                lower = stats[c].quantile(1-q)
                higher = stats[c].quantile(q)
                ax.scatter([lower, higher], [-i,-i], marker="|", color=player_colors[player])
                ax.plot(np.linspace(lower, higher, 2), [-i,-i], linewidth=2-n*0.3, color=player_colors[player], alpha=1-n*0.3)
            ax.set_title(c, y = -0.65)

    axes[1,4].annotate("mean, 90%, 95%, 99% quantile", xy =(7.2, -2),
                    xytext =(-21, -7),
                    arrowprops = dict(arrowstyle = "->"))

    text = fig.suptitle("Game stats confidence bars of different players", fontsize=16)
    lgd = axes[1,4].legend(bbox_to_anchor=(0.44, -1.2))
    plt.savefig(os.path.join(save_dir, "game_stats_confidence_bars_3_players.png"),
        bbox_extra_artists=(lgd,text), bbox_inches='tight')




# ===================== LAG CORRELATIONS =====================
def tweets_games_correlation_varying_lag(players: AllPlayers, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12,6))
    plt.subplots_adjust(hspace=0.35)

    # Plotting weak correlations with avg_positive
    weak_correlations = ['minutes', 'passes', 'foulsDrawn', 'tackles', 'blocks',  'duelsTotal', 'duelsWon', 'penaltiesScored']
    weak_offensive = plt.cm.Reds(np.linspace(0, 1, 2))[1:]
    weak_defensive = plt.cm.Blues(np.linspace(0, 1, 5))[1:]
    weak_passive = plt.cm.Greens(np.linspace(0, 1, 4))[1:]
    color_weak_correlatons = {
        'minutes' : weak_passive[0], 'passes' : weak_passive[1], 'foulsDrawn' : weak_passive[2],
        'tackles' : weak_defensive[0], 'blocks' : weak_defensive[1], 'duelsTotal' : weak_defensive[2], 'duelsWon' : weak_defensive[3],
        'penaltiesScored' : weak_offensive[0]}
    color_weak = list(color_weak_correlatons.values())
    players.Cristiano_Ronaldo.plot_games_with_tweets_correlation_varying_lag(
        'avg_positive', ax=axes[0,0], columns=weak_correlations, color = color_weak)
    players.Kevin_De_Bruyne.plot_games_with_tweets_correlation_varying_lag(
        'avg_positive', ax=axes[0,1], columns=weak_correlations, color = color_weak)
    players.Virgil_van_Dijk.plot_games_with_tweets_correlation_varying_lag(
        'avg_positive', ax=axes[0,2], columns=weak_correlations, color = color_weak)

    # Plotting interesting correlations with avg_positive
    interesting_correlatons = ['rating', 'shotsTotal', 'shotsOn', 'goals', 'assists']
    interesting_offensive = plt.cm.Reds(np.linspace(0, 1, 5))[1:]
    color_interesting_correlatons = {
        'rating' : 'lawngreen', 'shotsTotal' : interesting_offensive[0], 
        'shotsOn' : interesting_offensive[1], 'goals' : interesting_offensive[2], 
        'assists' : interesting_offensive[3]}
    color_interesting = list(color_interesting_correlatons.values())
    players.Cristiano_Ronaldo.plot_games_with_tweets_correlation_varying_lag(
        'avg_positive', ax=axes[1,0], columns=interesting_correlatons, color=color_interesting)
    players.Kevin_De_Bruyne.plot_games_with_tweets_correlation_varying_lag(
        'avg_positive', ax=axes[1,1], columns=interesting_correlatons, color=color_interesting)
    players.Virgil_van_Dijk.plot_games_with_tweets_correlation_varying_lag(
        'avg_positive', ax=axes[1,2], columns=interesting_correlatons, color=color_interesting)

    # General styling for all axes
    for ax in axes.flatten():
        ax.set_ylim([-0.4, 0.7])
        ax.grid()
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_ticklabels([])
        ax.hlines(0.2, -6, 6, linestyle='--', color='black', linewidth=1.5)
        ax.hlines(-0.2, -6, 6, linestyle='--', color='black', linewidth=1.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', direction='out', length=5, width=1)

    # Styling outer axes
    axes[0,0].yaxis.set_ticklabels([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    axes[1,0].yaxis.set_ticklabels([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    axes[0,0].set_title(f"Cristiano Ronaldo")
    axes[0,1].set_title(f"Kevin De Bruyne")
    axes[0,2].set_title(f"Virgil van Dijk")
    axes[0,1].get_legend().remove()
    axes[0,2].get_legend().remove()
    axes[1,1].get_legend().remove()
    axes[1,2].get_legend().remove()
    axes[0,0].xaxis.set_ticklabels([])
    axes[0,1].xaxis.set_ticklabels([])
    axes[0,2].xaxis.set_ticklabels([])

    # Inserting kde plots 
    for i1, (ax, player) in enumerate(zip([axes[1,0], axes[1,1], axes[1,2]], [players.Cristiano_Ronaldo, players.Kevin_De_Bruyne, players.Virgil_van_Dijk])):
        axin1 = ax.inset_axes([0.04, 1.05, 0.2, 0.20])
        axin2 = ax.inset_axes([0.04+1*(0.2 + 0.04), 1.05, 0.2, 0.20])
        axin3 = ax.inset_axes([0.04+2*(0.2 + 0.04), 1.05, 0.2, 0.20])
        axin4 = ax.inset_axes([0.04+3*(0.2 + 0.04), 1.05, 0.2, 0.20])
        for i2, (col, axin) in enumerate(zip(interesting_correlatons[1:], [axin1, axin2, axin3, axin4])):
            player.plot_col_kde(DataEnum.STATS, col, axin, color=color_interesting_correlatons[col])
            axin.spines[['top', 'left', 'right']].set_visible(False)
            if i1 == 0 and i2 == 0:
                axin.spines[['left']].set_visible(True)
                axin.set_ylabel("Density", fontsize=6)
                axin.set_yticks([])
            else:
                axin.get_yaxis().set_visible(False)
            for label in axin.get_xticklabels(): 
                label.set_fontsize(6)
            
    # Labels, legends and saving
    text1 = fig.text(0.5,0.04, "Tweet lag (Days)", ha="center", va="center", fontsize=10)
    text2 = fig.text(0.05,0.45, "Correlation (Avg positive)", ha="center", va="center", rotation=90, fontsize=10)
    lgd1 = axes[0,0].legend(bbox_to_anchor=(-0.2, 1.1))
    lgd2 = axes[1,0].legend(bbox_to_anchor=(-0.2, 0.5))
    plt.savefig(os.path.join(save_dir, "tweets_games_correlation_varying_lag.png"),  
        bbox_extra_artists=(text1, text2, lgd1, lgd2), bbox_inches='tight')




# ===================== Main functionality =====================
def init(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return AllPlayers()

def run_all_plots(save_dir):
    players = init(save_dir)

    # @Tor we need these 3
    # Add what you have, and then we save all plots together
    # If you have a lot of plots, perhaps add a folder to save_dir to make it cleaner
    basic_data_statistics(players, save_dir)
    number_of_tweets_for_each_player(players, save_dir)
    number_of_total_sentiments_for_each_player(players, save_dir)
    game_stats_confidence_bars_3_players(players, save_dir)
    tweets_games_correlation_varying_lag(players, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, default="Plots_new", required=False)
    args = parser.parse_args()
    run_all_plots(args.out)
