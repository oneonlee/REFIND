import numpy as np
import os
import pandas as pd
import urllib.request
import faiss
import time
from sentence_transformers import SentenceTransformer

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/19.%20Topic%20Modeling%20(LDA%2C%20BERT-Based)/dataset/abcnews-date-text.csv", filename="abcnews-date-text.csv")

df = pd.read_csv("abcnews-date-text.csv")
data = df.headline_text.to_list()

model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
task = "retrieval.passage"
encoded_data = model.encode(
                  data,
                  task=task,
                  prompt_name=task
               )
print('임베딩 된 벡터 수 :', len(encoded_data))

# index = faiss.IndexIDMap(faiss.IndexFlatL2(1024))
index = faiss.IndexIDMap(faiss.IndexFlatIP(1024))
index.add_with_ids(encoded_data, np.array(range(0, len(data))))

faiss.write_index(index, 'abc_news')

def search(query):
   t = time.time()
   query_vector = model.encode([query])
   k = 5
   top_k = index.search(query_vector, k)
   print('total time: {}'.format(time.time() - t))
   return [data[_id] for _id in top_k[1].tolist()[0]]

query = str(input())
results = search(query)

print('results :')
for result in results:
   print('\t', result)
