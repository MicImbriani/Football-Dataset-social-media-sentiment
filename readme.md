## Football-Twitter Dataset

This repository contains data and source code for a group project in the course: ***Data in the Wild: Wrangling and Visualising Data (Autumn 2022)***. 

The goal has been to create a dataset connecting the "on-the-field" performance of football players to Twitter engagement, e.g. tweets mentioning the players. Such dataset can be used for a wide varity of purposes. A few examples are:

- The engagement can act as a measure of "off-the-field" performance, giving insights about popularity and forecasted ticketsales
- Players that want to grow their social media platform can examine which "on-the-field" performance metrics that benefit engagement the most, and change their playstyle / tactics accordingly
- You can forecast and build prediction models about how social media engagement affects player performance and vice versa
- By connecting player relations to each other, you might be able to discover how a player transfer will affect social media engagement in a wider context


## Overview 
The repository is split in 2 main parts: `data` containing the dataset, and `src` containing the source code for working with the dataset. `data` is split in 2 parts, corresponding to two iterations of processing. Similarly `src` is split in 5 parts according to different stages of our work. Below we show the IO dependencies. We also provide a brief description of each main part.

!["io dependencies"](repository_io_dependencies.png)

### `data`
- `collected_with_some_processing`: Contains the collected data produced by `1 data_collecting`. This data is not meant to be used, but is kept for reproducability and transparency.
- `final`: Contains the data with its final processing produced by `3 additional_processing`. This data is ready to be used.

### `src`
- `1 data_collecting`: Contains code for collecting and processing tweets and game stats. This includes a sentiment analysis for each tweet based on a pretrained sentiment analysis model.
- `2 sentiment_evaluation`: Evaluates the performance of the sentiment analysis model. Contains code, instructions for annotation and evaluation.
- `3 additional_processing`: Contains code for reading the data in `collected_with_some_processing`, doing additional processing, and storing the results in `final`.
- `4 analysis_and_plots`: Contains code for reading the data in `final` and producing the plots shown in the report.
- `5 hydration`: Contains code for 'dehydrating' and 'rehydrating' Twitter data. Note that some scripts in the other parts only works if the Twitter data has been 'rehydrated'.

## Explanation of the dataset
***Note that the all Twitter data we share has been 'dehydrated'. That is, the text of the tweets has been replaced with the value 'dehydrated', to comply with Twitter terms and conditions. If users of this repository want access to the text (fx if they want to add their own sentiment analysis), they have to run a 'rehydration' script similar to what is shown in `5 hydration`.***

The dataset (`final`) is split in two directories: 
- `games`: Contains a csv file for each player, consisting of the players statistics in the games they have played in a 5 year period
- `tweets`: Contains two subdirections:
    - `all`: Contains a csv file for each player, consisting of up to 200 collected daily tweets with their sentiment analysis and likes, in a 5 year period
    - `aggregated`: Contains a csv file for each player, where each row is a daily aggregation of all tweets on that day

Below is an explanation of each column in the csv files

<br>

### `games`
Each row corresponds to a game played
| name | opponent | date | minutes | rating | shotsTotal | shotsOn | goals |
|----------|----------|----------|----------|----------|----------|----------|----------|
| player name | team name of the opponent | datetime of the game| minutesplayed on the field | a relative scoring of the players combined performance in the game | total shots made by the player | total shots on goal made by the player | goals made by the player |

| assists | passes | tackles | blocks | duelsTotal | duelsWon | foulsDrawn | penaltiesScored |
|----------|----------|----------|----------|----------|----------|----------|----------|
|assists made by the player| passes made by the player | tackles made by the player | blocks made by the player | amount of duels the player was part of | amount of duels won | number of times the player was fouled while in possesion of the ball | number of penalties scored |


<br>

### `tweets/all`
Each row corresponds to a single tweet. A maximum of 200 most relevant tweets are collected on a given day, based on Twitters 'relevancy' metric.
| player_name | tweet_id | text | date | likes | negative | neutral | positive |
|----------|----------|----------|----------|----------|----------|----------|----------|
| name of the player (same as in games) | id of the tweet in the Twitter API | textual content (is 'dehydrated') | datetime of the tweet | amount of likes the tweet received on Twitter | proportion of negative sentiment in the tweet (according to the pretrained model) | proportion of neutral sentiment in the tweet (according to the pretrained model) | proportion of positive sentiment in the tweet (according to the pretrained model) |

<br>

### `tweets/aggregated`
Each row corresponds to a daily aggregation of the tweets in `tweets/all`. That is, tweets made on the same day are aggregated/condensed into a single row.
| player_name | date | total_tweets | avg_likes | avg_negative | avg_neutral | avg_positive | 
|----------|----------|----------|----------|----------|----------|----------|
| name of the player (same as in games) | datetime of the day | amount of tweets aggregated | average likes of the  aggregated tweets | average proportion of negative sentiment of the aggregated tweets | average proportion of neutral sentiment of the aggregated tweets | average proportion of positive sentiment of the aggregated tweets |

| total_negative | total_neutral | total_positive | count |
|----------|----------|----------|----------|
| the amount of tweets where the proportion of negative sentiment is largest | the amount of tweets where the proportion of neutral sentiment is largest | the amount of tweets where the proportion of positive sentiment is largest | the total amount of tweets that could have been collected for the day (can be over 200)  | 

<br>

See the report for a further explanation and analysis of the dataset