# Download fi_wiki dump
wget https://dumps.wikimedia.org/fiwiki/latest/fiwiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess fi_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/fiwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/fi_wiki_corpus.jsonl \
    --num_workers 16