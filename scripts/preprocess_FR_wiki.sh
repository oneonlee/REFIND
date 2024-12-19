# Download fr_wiki dump
wget https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess fr_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/frwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/fr_wiki_corpus.jsonl \
    --num_workers 16