# slightly improved performace, cleaner tokens
from utils import read, write
import pandas as pd
import numpy as np
import re
import langdetect as ld
from utils import read
import swifter 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

def preprocess_part_1(text):
    text = text.encode('ascii', errors='ignore').decode() 
    text = text.lower()
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\w+…|…", "", text) 
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text) 
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[0-9]+", '', text)

    return text

def detect_english(text): # flag column
    try:
        return ld.detect(text) == 'en'
    except:
        return False
    
def preprocess_part_2(text):
    stop_words = stopwords.words('english')
    wn = WordNetLemmatizer()
    text = word_tokenize(str(text))
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if word.isalpha()]
    text = [wn.lemmatize(word) for word in text]

    return text

def main(df): # 
    df['clean_p1'] = df['Tweet'].swifter.apply(lambda x: preprocess_part_1(x))
    df['eng'] = df['clean_p1'].swifter.apply(lambda x: detect_english(x))
    indexNotEng = df[(df['eng'] != True)].index #drop rows flagged as False which means not english
    df.drop(indexNotEng , inplace=True)
    df['cleaned'] = df['clean_p1'].swifter.apply(lambda x: preprocess_part_2(x))

if __name__ == "__main__":
    df = read('tweets_fact_table', limit = 1200000)
    main(df)
    write(df, "tweets_fact_table")

    
    
    

