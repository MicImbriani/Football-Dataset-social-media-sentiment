# This file does additional processing to the tweets in data/collected_with_some_processing/tweets
# The processed data is stored in data/final
# In particular the following procecessing is done:
# - 'tweet_date' is renamed to 'date' in in all-tweets to keep naming consistent
# - null / empty values in aggregated tweets are replaced with 0

# Besides that, we found an error during data collection, where if some tweets containing specific symbols,
# then it would mess with the column/row structure of the csv format. 
# To fix this issue, we have removed affected rows (only very few occurrences)


import os
import pandas as pd

def process_player_tweets_all(read_file: str, dest_file: str):
    df = pd.read_csv(read_file, parse_dates=['tweet_date'], lineterminator='\n')
    df = df.rename(columns={'tweet_date': 'date'})
    df.date = pd.to_datetime(df.date, errors='coerce')
    df.dropna(inplace=True)
    df.to_csv(dest_file, index=False)

def process_player_tweets_aggregated(read_file: str, dest_file: str):
    df = pd.read_csv(read_file, parse_dates=['date'])
    df.fillna(0, inplace=True)
    df.to_csv(dest_file, index=False)


if __name__ == '__main__':
    read_dir = '../../data/collected_with_some_processing/tweets'
    dest_dir = '../../data/final/tweets'
    players = ['Cristiano Ronaldo', 'Erling Haaland', 'Kevin De Bruyne',
                'Kylian Mbappé', 'Lionel Messi', 'Neymar', 'Riyad Mahrez', 
                'Robert Lewandowski', 'Sadio Mané', 'Virgil van Dijk']

    for player in players:
        read_file_all = os.path.join(read_dir, 'all', f'{player}.csv')
        dest_file_all = os.path.join(dest_dir, 'all', f'{player}.csv')
        process_player_tweets_all(read_file_all, dest_file_all)

        read_file_aggregated = os.path.join(read_dir, 'aggregated', f'{player}.csv')
        dest_file_aggregated = os.path.join(dest_dir, 'aggregated', f'{player}.csv')
        process_player_tweets_aggregated(read_file_aggregated, dest_file_aggregated)
