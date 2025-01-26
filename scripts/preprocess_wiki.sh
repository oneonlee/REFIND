for lang in ar ca cs de en es eu fa fi fr it zh; do
    wiki_url="https://dumps.wikimedia.org/${lang}wiki/latest/${lang}wiki-latest-pages-articles.xml.bz2"
    # e.g., https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

    # Download wiki dump
    wget $wiki_url -P retriever

    # Preprocess wiki dump
    python retriever/preprocess_wiki.py \
        --dump_path retriever/${lang}wiki-latest-pages-articles.xml.bz2  \
        --save_path retriever/${lang}_wiki_corpus.jsonl \
        --num_workers 16
done