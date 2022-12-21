from utils import client_id, client_secret, user_agent, check_if_valid, write
import pandas as pd
import praw

reddit = praw.Reddit(client_id=client_id, 
client_secret=client_secret, user_agent=user_agent)

all_comments = []
subreddit = reddit.subreddit('startups')
for post in subreddit.hot(limit=None):
    if not post.stickied:
        post.comments.replace_more(limit=0)
        comments = post.comments.list() 
        for comment in comments:
            all_comments.append([comment.id, comment.body])

df = pd.DataFrame(all_comments, columns=['ID', 'comment'])

if check_if_valid(df):
    print("No problems")
    
write(df, "reddit_fact_table")



