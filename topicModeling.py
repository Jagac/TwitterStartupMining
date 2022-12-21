import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import umap.umap_ as umap
import hdbscan
from hyperopt import fmin, tpe, hp, space_eval, Trials, partial
import matplotlib.pyplot as plt

df = pd.read_csv('sentiments.csv')
tweets = list(df["clean_p1"].values)
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)

def embed(input):
  return model(input)

embeddings = embed(tweets)
print(embeddings.shape)

def get_topics(embeddings, n_neighbors, n_components, min_cluster_size):
    #References:
    #https://umap-learn.readthedocs.io/en/latest/document_embedding.html
    #https://umap-learn.readthedocs.io/en/latest/parameters.html
    #https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    dim_reduce = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine').fit_transform(embeddings)
    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,metric='euclidean', cluster_selection_method='eom').fit(dim_reduce)

    return clusters

def cost_function(clusters, probability = 0.05):
    # https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html
    # https://towardsdatascience.com/how-to-cluster-in-high-dimensions-4ef693bacc6
    count = len(np.unique(clusters.labels_))
    total = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < probability)/total)
    
    return count, cost

"""
clusters= get_topics(embeddings, n_neighbors = 25, n_components = 15, min_cluster_size = 10)
labels, cost = cost_function(clusters)
print(cost)
"""

def optimization_function(embeddings, params, lower_bound, upper_bound):
    # https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

    clusters = get_topics(embeddings, n_neighbors = params['n_neighbors'], n_components = params['n_components'], min_cluster_size = params['min_cluster_size'])
    count, cost = cost_function(clusters, probability = 0.05)
    if (count < lower_bound) | (count > upper_bound):
        penalty = 0.15 
    else:
        penalty = 0
    loss = cost + penalty
    results = {'loss': loss, 'number of labels': count}

    return results

def bayesian_optimization(embeddings, search_space, lower_bound, upper_bound, total_evaluations):
    #https://github.com/hyperopt/hyperopt

    trials = Trials()
    fmin_optimize = partial(optimization_function, embeddings = embeddings, lower_bound=lower_bound, upper_bound=upper_bound)
    best = fmin(fmin_optimize, search_space = search_space, algo=tpe.suggest, total_evaluations = total_evaluations, trials=trials)
    best_params = space_eval(search_space, best)
    #print(best_params)
    best_clusters = get_topics(embeddings, n_neighbors = best_params['n_neighbors'], n_components = best_params['n_components'], min_cluster_size = best_params['min_cluster_size'])
    
    return best_params, best_clusters, trials


def plot_results(embeddings, clusters, n_neighbors=15, min_dist=0.1):
    reduced_dim = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist = min_dist).fit_transform(embeddings)
    result = pd.DataFrame(reduced_dim, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color = 'lightgrey')
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, cmap='jet')
    plt.colorbar()
    plt.savefig("clusterresutls.png")

search_space = { "n_neighbors": hp.choice('n_neighbors', range(3, 50)),
                 "n_components": hp.choice('n_components', range(3, 50)),
                 "min_cluster_size": hp.choice('min_cluster_size', range(3, 50))}

lower_bound = 25
upper_bound = 50
max_eval = 50
best_params, best_clusters, trials = bayesian_optimization(embeddings, search_space=search_space, lower_bound=lower_bound, upper_bound=upper_bound, max_eval=max_eval)
print(trials.best_trial)

plot_results(embeddings, best_clusters)
cluster_dict = {'cluster label': best_clusters}

df_2 = df.copy()
for key, value in cluster_dict.items():
    df_2[['ID', 'Tweet', 'cleaned', 'sentiment']] = value.labels_

df_2.to_csv('finalDf.csv')
#df.to_csv("test.csv")