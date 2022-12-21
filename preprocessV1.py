# major issues with performance and cannot detect underscores
import re
import numpy as np
import pandas as  pd
from pprint import pprint
import gensim
from gensim.utils import simple_preprocess
from utils import read, write
import spacy
import langdetect as ld
from nltk.corpus import stopwords
import nltk 
from nltk import word_tokenize
import string
#nltk.download('punkt')
stop_words = stopwords.words('english')

df = read('tweets_fact_table', limit = 10000)

def remove_non_ascii(text): #weird letters
    return text.encode('ascii', errors='ignore').decode()

def detect_english(text): # flag column
    try:
        return ld.detect(text) == 'en'
    except:
        return False

def remove_general(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002700-\U000027BF"  
                            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub('\S*@\S*\s?', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub("\w+…|…", "", text) 
    text = re.sub("(?<=\w)-(?=\w)", "", text)
    text = re.sub("(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub("'", "", text)
    return text

def sent_to_words(text):
    "".join([word for sublist in [word_tokenize(x) if '_' not in x else [x] 
                       for x in text] for word in sublist])
    return text

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

df['ascii'] = df['Tweet'].swifter.apply(lambda x: remove_non_ascii(x))
df['eng'] = df['ascii'].swifter.apply(lambda x: detect_english(x))
indexNotEng = df[(df['eng'] != True)].index #drop rows flagged as False which means not english
df.drop(indexNotEng , inplace=True)
df['cleanedup'] = df['ascii'].swifter.apply(lambda x: remove_general(x))
df['punct'] = df['cleanedup'].swifter.apply(lambda x: remove_punct(x))


data = df.cleanedup.values.tolist()  

data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)

# initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

with open("20k.txt", 'w') as output:
    for row in data_lemmatized:
        output.write(str(row))




