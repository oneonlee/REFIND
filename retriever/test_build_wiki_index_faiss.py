import os

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


df = pd.read_csv("abcnews-date-text.csv") # TODO
data = df.headline_text.to_list() # TODO

model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
task = "retrieval.passage"
encoded_data = model.encode(
                  data,
                  task=task,
                  prompt_name=task
               )
print("number of embedded vectors :", len(encoded_data))

index = faiss.IndexIDMap(faiss.IndexFlatL2(1024))
index.add_with_ids(encoded_data, np.array(range(0, len(data))))
faiss.write_index(index, 'enwiki-latest-pages-articles.index')


################################## Search ##################################
data = df.headline_text.to_list() # TODO
index = faiss.read_index('enwiki-latest-pages-articles.index')

def search(query, k=5):
   query_embedding = model.encode(
                        [query],
                        task='retrieval.query',
                        prompt_name='retrieval.query'
                     )
   top_k = index.search(query_embedding, k)
   return [data[_id] for _id in top_k[1].tolist()[0]]

query = "What is the capital of France?"
results = search(query)

print('results :')
for result in results:
   print('\t', result)
