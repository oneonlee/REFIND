# Download zh_wiki dump
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess zh_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/zhwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/zh_wiki_corpus.jsonl \
    --num_workers 16