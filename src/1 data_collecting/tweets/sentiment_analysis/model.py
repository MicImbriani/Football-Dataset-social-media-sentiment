import pathlib
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import pandas as pd
import os

class SentimentModel:
    def __init__(self):
        ROOT = pathlib.Path(__file__).parent.resolve()
        MODEL_DIR = os.path.join(ROOT, 'models')
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL)
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        if not os.path.exists(MODEL_PATH):
            print('downloading sentiment model')
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
            self.model.save_pretrained(MODEL_PATH)
        else:
            print('sentiment model has already been downloaded')
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

            """
        0 -> negative
        1 -> neutral
        2 -> positive
        """
        self.labels = ['negative', 'neutral', 'positive']


    # Preprocess text (username and link placeholders)
    def __preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)


    def __get_attitude(self, input: str) -> pd.Series:
        preprocessed = self.__preprocess(input)
        encoded_input = self.tokenizer(preprocessed, return_tensors='pt', truncation=True, max_length=512)
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        sentiment = softmax(scores)
        return pd.Series({self.labels[0]: sentiment[0], self.labels[1]: sentiment[1], self.labels[2]: sentiment[2]})


    def add_attitudes(self, df: pd.DataFrame, col: str = 'text') -> pd.DataFrame:
        if df.empty:
            opinions = pd.DataFrame(columns=self.labels)
        else:
            opinions = df.apply(lambda x: self.__get_attitude(x[col]), axis=1)
        return pd.concat([df, opinions], axis=1, join='inner')
