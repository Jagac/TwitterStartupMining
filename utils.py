import pymysql
from sqlalchemy import create_engine
import pandas as pd

# for reddit api
client_id = "W7aYXChi5MSt-atHVDqcYg"
client_secret = "YGWp89UXb_cp1wnfyJSKiz4T2_NAlA"
user_agent = "jagacscrape"

def check_if_valid(df: pd.DataFrame) -> bool:
    # Check if dataframe is empty
    if df.empty:
        print("No new data")
        return False 
    # Primary key check
    if pd.Series(df["ID"]).is_unique:
        pass
    else:
        raise Exception("Primary key check is violated")
    # Check for nulls
    if df.isnull().values.any():
        raise Exception("Null values found")

    return True

def double_check(df: pd.DataFrame) -> bool:
    if len(df) == len(df['ID'].unique()):
        pass
    else:
        raise Exception("Primary key is violated")
    
    return True

def read(table, limit):
    engine = create_engine("mysql+pymysql://root:jagacjecar123@localhost/twitterstartups")
    df = pd.read_sql_query(f'SELECT * FROM {table} LIMIT {limit}', con= engine)
    if double_check(df):
        print("Valid")
          
    return df

def write(df, table):
    engine = create_engine("mysql+pymysql://root:jagacjecar123@localhost/twitterstartups")
    df.to_sql(f'{table}', con = engine, if_exists = 'append',index = False, chunksize = 1000)

#nltk.download('omw-1.4')
#nltk.download('wordnet')


