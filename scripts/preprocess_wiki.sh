python retriever/preprocess_wiki.py \
    --dump_path retriever/enwiki-latest-pages-articles.xml.bz2  \
    --save_path retriever/en_wiki_corpus_w100.jsonl \
    --num_workers 16 \
    --chunk_by 100w
2