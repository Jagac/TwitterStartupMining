import pandas as pd
from utils import check_if_valid, write
import snscrape.modules.twitter as sntwitter

keyword = "#startups"
max_results = 1200000
tweet_data = []

# about 100k tweets per hour
for i, tweets in enumerate(sntwitter.TwitterSearchScraper('{}'.format(keyword)).get_items()):
    if i > max_results:
        break
    tweet_data.append([tweets.id, tweets.content])
    print(i)
    
df = pd.DataFrame(tweet_data, columns=['ID','Tweet'])

if check_if_valid(df):
    print("No problems")

write(df, "tweets_fact_table")

