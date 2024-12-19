# Download de_wiki dump
wget https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess de_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/dewiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/de_wiki_corpus.jsonl \
    --num_workers 16