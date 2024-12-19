# Download sv_wiki dump
wget https://dumps.wikimedia.org/svwiki/latest/svwiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess sv_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/svwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/sv_wiki_corpus.jsonl \
    --num_workers 16