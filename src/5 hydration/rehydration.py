# This file contains a simple example of how to 'rehydrate' all twitter data
# That is, it fetches all the 'text' from the Twitter API based on the id's
# It makes a tweepy request for each row.
# Hence you can do it more efficiently if you want to avoid getting rate limited

import tweepy
import pandas as pd
import os

client = tweepy.Client(bearer_token='insert-your-token-here')

def rehydrate(path: str):
    df = pd.read_csv(path, lineterminator='\n') 

    # Makes a request per row (can be done with more rows per request)
    for i, row in df.iterrows():
        tweet = client.get_tweet(id=row['tweet_id'])
        df.at[i,'text'] = tweet.data['text']

    df.to_csv(path, index=False)
    

if __name__ == '__main__':
    # Rehydrate data directory
    data_dir_1 = "../../data/collected_with_some_processing/tweets/all"
    for filename in os.listdir(data_dir_1):
        rehydrate(os.path.join(data_dir_1, filename))

    data_dir_2 = "../../data/final/tweets/all"
    for filename in os.listdir(data_dir_2):
        rehydrate(os.path.join(data_dir_2, filename))