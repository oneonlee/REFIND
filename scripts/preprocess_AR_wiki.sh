# Download ar_wiki dump
wget https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess ar_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/arwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/ar_wiki_corpus.jsonl \
    --num_workers 16

    