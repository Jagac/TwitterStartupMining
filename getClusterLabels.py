import collections
import pandas as pd
import spacy 
import swifter

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("dataFinal.csv")
print(df.columns)

def get_group(df, label, category):
    group = df[df[label]==category]
    return group 

def most_common(list1, words):
    counter=collections.Counter(list1)
    most_frequent = counter.most_common(words)
    return most_frequent

def labels(docs):
    verbs = [], object1 = [], nouns = [], adject = []
    verb = ''
    dobj = ''
    noun1 = ''
    noun2 = ''

    for i in range(len(docs)):
        doc = nlp(docs[i])

        for token in doc:
            if token.is_stop==False:
                if token.dep_ == 'ROOT':
                    verbs.append(token.text.lower())
                elif token.dep_=='OBJ':
                    object1.append(token.lemma_.lower())
                elif token.pos_=='NOUN':
                    nouns.append(token.lemma_.lower())
                elif token.pos_=='ADJ':
                    adject.append(token.lemma_.lower())
    
    if len(verbs) > 0:
        verb = most_common(verbs, 2)[0][0]
    if len(object1) > 0:
        object1 = most_common(object1, 1)[0][0]
    if len(nouns) > 0:
        noun1 = most_common(nouns, 2)[0][0]
    if len(set(nouns)) > 1:
        noun2 = most_common(nouns, 1)[1][0]

    labels = [verb, object1]
    
    for word in [noun1, noun2]:
        if word not in labels:
            labels.append(word)
    label = '_'.join(labels)
    
    return label


def summary(df, col):
    labels = df[col].unique()
    label_dict = {}
    for label in labels:
        current_label = list(get_group(df, col, label)['cluster_label'])
        label_dict[label] = labels(current_label)
        
    summary_df = (df.groupby(col)['cluster label'].count().sort_values('count', ascending=False))
    summary_df['label'] = summary_df.swifter.apply(lambda x: label_dict[x[col]], axis = 1)
    
    return summary_df

cluster_summary = summary(df, 'cluster label')
cluster_summary.to_csv('labeledclusters.csv')

