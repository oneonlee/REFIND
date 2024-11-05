python retriever/preprocess_wiki.py \
    --dump_path retriever/enwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/en_wiki_corpus.jsonl \
    --num_workers 9 \
    --chunk_by 100w