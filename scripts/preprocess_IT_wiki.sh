# Download it_wiki dump
wget https://dumps.wikimedia.org/itwiki/latest/itwiki-latest-pages-articles.xml.bz2 -P retriever

# Preprocess it_wiki dump
python retriever/preprocess_wiki.py \
    --dump_path retriever/itwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/it_wiki_corpus.jsonl \
    --num_workers 16