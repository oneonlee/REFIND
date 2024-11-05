python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/multilingual-e5-large \
    --corpus_path retriever/en_wiki_corpus.jsonl \
    --save_dir retriever/ \
    # --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --sentence_transformer \
    --faiss_gpu \
    --faiss_type Flat 