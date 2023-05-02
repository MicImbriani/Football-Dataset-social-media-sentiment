# This file 'dehydrates' all twitter data according to Twitters content policy
# That is, it sets all the 'text' columns of our Twitter data to be "dehydrated"
# The id's are kept, so users of our data can 'rehydrate' it and add the text back
# Note that the files in sentiment evaluation is already dehydrated, 
# as the text is replaced by public facing Twitter links

import pandas as pd
import os

def dehydrate(path: str):
    df = pd.read_csv(path, lineterminator='\n') 
    df['text'] = "dehydrated"
    df.to_csv(path, index=False)

if __name__ == '__main__':
    # Dehydrate data directory
    data_dir_1 = "../../data/collected_with_some_processing/tweets/all"
    for filename in os.listdir(data_dir_1):
        dehydrate(os.path.join(data_dir_1, filename))

    data_dir_2 = "../../data/final/tweets/all"
    for filename in os.listdir(data_dir_2):
        dehydrate(os.path.join(data_dir_2, filename))
