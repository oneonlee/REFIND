# Download es_wiki dump
wget https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess es_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/eswiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/es_wiki_corpus.jsonl \
    --num_workers 16