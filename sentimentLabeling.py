import pandas as pd
from preprocessV2 import detect_english, preprocess_part_1, preprocess_part_2
import swifter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
from collections import Counter
# Model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

# random sampling
num_records = 1200000
sample = 500000 
skip = sorted(random.sample(range(num_records), num_records - sample))
df = pd.read_csv("tweets.csv", skiprows = skip)

df['clean_p1'] = df['Tweet'].swifter.apply(lambda x: preprocess_part_1(x))
df['eng'] = df['clean_p1'].swifter.apply(lambda x: detect_english(x))
indexNotEng = df[(df['eng'] != True)].index #drop rows flagged as False which means not english
df.drop(indexNotEng , inplace=True)
df['cleaned'] = df['clean_p1'].swifter.apply(lambda x: preprocess_part_2(x))

tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

def sentiment_score(review):
  tokens = tokenizer.encode(review, return_tensors='pt')
  result = model(tokens)
  return int(torch.argmax(result.logits))

df['sentiment'] = df['clean_p1'].swifter.apply(lambda x: sentiment_score(x[:512]))
print(Counter(" ".join(df["clean_p1"]).split()).most_common(1000))

df[["ID", "Tweet", "clean_p1", "cleaned", "sentiment"]].to_csv("sentiments.csv")