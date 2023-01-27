# What Makes a Successful Startup?

## Description

This project scrapes about 1 million tweets from #startups, which are then stored as a table inside a MySQL database. The tweets are then preprocessed using 
one of the preprocess scripts. To get a better understanding of the topics behind all these tweets, a pretrained Tensorflow model can be used to vectorize the sentences. 
The vector dimensions are then reduced by UMAP and clustered using HDBSCAN. The roBERTa model was used to label tweet sentiments. 

## Further references

* For scraping tweets without an API: https://github.com/JustAnotherArchivist/snscrape
* Pretrained models: https://tfhub.dev/google/universal-sentence-encoder/4, https://huggingface.co/docs/transformers/model_doc/roberta
* UMAP docs: https://umap-learn.readthedocs.io/en/latest/
* HDBSCAN docs: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
