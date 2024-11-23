# Download en_wiki dump
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess en_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/enwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/en_wiki_corpus.jsonl \
    --num_workers 16