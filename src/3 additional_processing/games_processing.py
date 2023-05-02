# This file does additional processing to the game data in data/collected_with_some_processing/games
# The processed data is stored in data/final
# Note that this processing is only done for the players we use in the analysis
# In particular the following procecessing is done:
# - 'date' columns is changed to have datetime type
# - "None" strings are replaced with 0
# - Every row with a non-number rating is removed
# - Player names are changed to their full versions rather than abbreviations


import os
import pandas as pd

# Reads player games and cleans up the data as appropriate
def process_player_games(player: str, read_dir: str, dest_dir: str):
    root_stats_dir = os.path.join(read_dir, player)
    save_file = os.path.join(dest_dir, f'{player}.csv')
    for filename in os.listdir(root_stats_dir): # Take non-raw columns
        if filename.endswith(".csv") and "RAW" not in filename:
            player_games = pd.read_csv(os.path.join(root_stats_dir, filename), parse_dates=['date'])
            player_games['date'] = pd.to_datetime(player_games['date'], utc = True)
            player_games = player_games.replace("None", 0)
            player_games['rating'] = pd.to_numeric(player_games['rating'], errors = 'coerce')
            player_games.dropna(inplace = True)
            player_games = player_games.sort_values(by=['date'])
            player_games = player_games.iloc[:,1:] # Drop first column since its just the row number
            player_games.name = player
            player_games.astype({
                'rating'     : 'Float64',
                'shotsTotal' : 'Float64',
                'shotsOn' : 'Float64',
                'goals' : 'Float64',
                'assists' : 'Float64',
                'tackles' : 'Float64',
                'blocks' : 'Float64',
                'duelsTotal' : 'Float64',
                'duelsWon' : 'Float64',
                'shotsTotal' : 'Float64',
                'foulsDrawn' : 'Float64',
                })
            player_games.to_csv(save_file, index=False)



if __name__ == '__main__':
    read_dir = '../../data/collected_with_some_processing/games'
    dest_dir = '../../data/final/games'
    players = ['Cristiano Ronaldo', 'Erling Haaland', 'Kevin De Bruyne',
                'Kylian Mbappé', 'Lionel Messi', 'Neymar', 'Riyad Mahrez', 
                'Robert Lewandowski', 'Sadio Mané', 'Virgil van Dijk']
    for player in players:
        process_player_games(player, read_dir, dest_dir)
