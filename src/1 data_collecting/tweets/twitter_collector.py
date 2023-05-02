# This file collects all the twitter data for each player
# Specifically the following is collected
# - The 200 most relevant tweets (according to twitter metric) for each day between 2017-08-01 and 2022-10-17
# - Likes for each tweet
# - A count of how many tweets could have been collected for each day

# We also do processing at the same time using a model trained for sentiment analysis on each tweet
# 2 files are produced for the player in the two folders:
# - all: Each row is a tweet about the player, with sentiment, datetime, id and like
# - aggregated: Each row is an aggregation of all of the 200 tweets for that day

# As the twitter API often fails (5xx errors), tweepy has bugs, and the collection takes a very long time,
# we have structured the data collection in a robust way to continue from the last row, rather than starting over

# Finally, note that this script stores raw tweets in the text column stored in the "all" folder. 
# Raw tweets are not allowed to be shared, so we remove this column in a later processing step

from genericpath import exists
from dateutil import parser
import pandas as pd
import tweepy
import time
import datetime
from operator import itemgetter
from sentiment_analysis.model import SentimentModel
import pathlib
import os

BEARER_TOKEN = "insert-your-token-here"
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
df_all_cols = ['player_name', 'tweet_id', 'text', 'tweet_date', 'likes', 'negative', 'neutral', 'positive']
df_agg_cols = ['player_name', 'date', 'total_tweets', 'avg_likes', 'avg_negative', 'avg_neutral', 'avg_positive', 'total_negative', 'total_neutral', 'total_positive']



# This methods collects 'amount' number of tweets based on relevancy between a (daily) timeframe.
# A buffer of +50 is used, since Tweepy often returns less than what we ask for without it
def get_tweets(player_name: str, query: str, start_time: datetime, end_time: datetime, amount: int) -> pd.DataFrame:
    tweets = []

    search_result = client.search_all_tweets(query=query,
                              tweet_fields=['created_at', 'public_metrics'],
                              start_time=start_time,
                              sort_order='relevancy',
                              end_time=end_time, max_results=amount + 50) # a buffer of 50
    
    if search_result.data is not None:
        for tweet in search_result.data[:amount]: # discard the buffer tweets
            series = pd.Series({'player_name': player_name, 'tweet_id': tweet['id'],
                                    'text': tweet['text'], 'tweet_date': tweet['created_at'], 'likes': tweet['public_metrics']['like_count']})
            tweets += [series]
        
    return pd.DataFrame(tweets, columns=['player_name', 'tweet_id', 'text', 'tweet_date', 'likes'])


# Returns true if the dataframe already has a row with this date.
# Is used to avoid starting over if the collection process fails and needs to be restarted
def df_contains_date(df, query_start_time):
    contains = df[df.date == query_start_time].shape[0] > 0
    if contains:
        print(f'SKIPPING - Tweets for {query_start_time} already exists')
    else:
        print(f'ADDING - Tweets for {query_start_time}')
    return contains

# Adds a count column to aggregated tweets.
# This column contains all the tweets that could have been collected for that day (e.g. can be over 200)
def add_counts_to_aggregated(start_time, end_time, query, aggregated_df, path):
    for tweet_count in tweepy.Paginator(
                                client.get_all_tweets_count, 
                                query=query,
                                start_time = start_time,
                                granularity = 'day',
                                end_time = end_time + datetime.timedelta(days=1)).flatten(limit=10000):
        count = tweet_count['tweet_count']
        date = tweet_count['start']
        aggregated_df.loc[aggregated_df.date == date, "count"] = count
    aggregated_df.to_csv(path, index=False)


# This method collects 'amount' daily tweets between start and stop days, if they have not already been collected
# It also runs the sentiment model on each tweet, and stores aggregated information in another dataframe
def collect_and_append(player_name, query, start, stop, amount, path_all, df_all, path_aggregated, model):
    player_rows = df_all[df_all.player_name == player_name]
    query_start_time = start
    while query_start_time <= stop:
        query_end_time = query_start_time + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
        if not df_contains_date(player_rows, query_start_time):
            time.sleep(1) # Hack, since tweepy thinks we reach the rate limit too early (tweepy bug)
            all_tweets = get_tweets(player_name, query, query_start_time, query_end_time, amount)
            all_tweets = model.add_attitudes(all_tweets)

            # Count the total amount of the highest sentiment scores
            dominant_sentiment = [0,0,0] # negativ, neutral, positive
            for _, row in all_tweets.iterrows():
                i, _ = max(enumerate([row["negative"], row["neutral"], row["positive"]]), key=itemgetter(1))
                dominant_sentiment[i] += 1

            count = pd.DataFrame({
                "player_name" : [player_name], 
                "date" : [query_start_time], 
                "total_tweets" : [all_tweets.shape[0]],
                "avg_likes" : [all_tweets.likes.mean()],
                "avg_negative_score" : [all_tweets.negative.mean()],
                "avg_neutral_score" : [all_tweets.neutral.mean()],
                "avg_positive_score" : [all_tweets.positive.mean()],
                "total_negative" : [dominant_sentiment[0]],
                "total_neutral" : [dominant_sentiment[1]],
                "total_positive" : [dominant_sentiment[2]],
                })
            count.to_csv(path_all, mode='a', index=False, header=False)
            if not all_tweets.empty:
                all_tweets.to_csv(path_aggregated, mode='a', index=False, header=False)
        query_start_time = query_end_time + datetime.timedelta(seconds=1)


# Reads and/or creates a dataframe containing all tweets
def get_all_df(path: str):
    if exists(path):
        return pd.read_csv(path, parse_dates=['tweet_date'], engine='python')
    df = pd.DataFrame(columns=df_all_cols)
    df.to_csv(path, index=False)
    return df

# Reads and/or creates a a dataframe containing aggregated tweets
def get_aggregated_df(path: str):
    if exists(path):
        return pd.read_csv(path, parse_dates=['date'])
    df = pd.DataFrame(columns=df_agg_cols)
    df.to_csv(path, index=False)
    return df

# Entire pipeline for collecting and storing tweets for a player
def collect_tweets_for_player(player_name, query, save_dir):
    start_time = parser.parse('2017-08-01 00:00:00+00:00')
    end_time = parser.parse('2022-10-17 00:00:00+00:00')
    amount = 200

    path_all_tweets = os.path.join(save_dir, 'all', f'{player_name}.csv')
    path_aggregated_tweets = os.path.join(save_dir, 'aggregated', f'{player_name}.csv')
    
    print('Getting Dataframes')
    df_all = get_all_df(path_all_tweets)
    df_aggregated_tweets = get_aggregated_df(path_aggregated_tweets)

    print("Creating sentiment model")
    model = SentimentModel()

    print(f'Getting tweets between {start_time} and {end_time}')
    collect_and_append(player_name, query, start_time, end_time, amount, 
                        path_all_tweets, df_all, path_aggregated_tweets, model)
    add_counts_to_aggregated(start_time, end_time, query, df_aggregated_tweets, path_aggregated_tweets)




if __name__ == '__main__':
    # General strategy:
    # We want to limit the amount of false positives while also have a general enough querty to get enough tweets
    # For players that dominate the search-space for their lastname, we can just search for the lastname
    # Other players will be searched by at least 2 names
    twitter_queries = {
        "Erling Haaland" : "haaland (erling OR braut) lang:en -is:retweet", # People tweet with his middle name also
        "Cristiano Ronaldo" : '"cristiano ronaldo" lang:en -is:retweet', # Most tweets regarding ronaldo is either 'ronaldo' or 'cristiano ronaldo' but we don't want to risk "false" ronaldos
        "Kevin De Bruyne" : '"de bruyne" lang:en -is:retweet', # Most tweets also contain "Kevin", but some do not. This might give false "de bruyne"s but it is unlikely
        "Kylian Mbappé" : "mbappe lang:en -is:retweet", # "Mbappe" is likely to have no overlaps, so we get more tweets by omitting "Kylian"
        "Lionel Messi" : "messi lang:en -is:retweet", # "Messi" is likely to have no overlaps / dominate, so we get more tweets by omitting "Lionel"
        "Neymar" : "neymar lang:en -is:retweet", # Neymar is almost only referrenced by one name
        "Riyad Mahrez" : "riyad mahrez lang:en -is:retweet", # Enforce both names, order does not matter 
        "Robert Lewandowski" : "robert lewandowski lang:en -is:retweet", # Enforce both names, order does not matter 
        "Sadio Mané" : "sadio mane lang:en -is:retweet", # Enforce both names, order does not matter 
        "Virgil van Dijk" : 'virgil "van Dijk" lang:en -is:retweet', # Enforce both names, order does not matter 
        }

    save_dir = "../../../data/colleced_with_some_processing/tweets"

    for player_name in twitter_queries.keys():
        print(f"Collecting for {player_name}")
        # Keeps retrying getting tweets if twitter gives 5xx server error (occurs quite often at the moment)
        while True:
            try:
                collect_tweets_for_player(player_name, twitter_queries[player_name], save_dir)
            except tweepy.TwitterServerError:
                print(f"Twitter server error for {player_name}")
                continue
            break
        print("\n")
    
