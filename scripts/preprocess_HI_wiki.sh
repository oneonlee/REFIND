# Download hi_wiki dump
wget https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2 -P retriever


# Preprocess hi_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/hiwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/hi_wiki_corpus.jsonl \
    --num_workers 16