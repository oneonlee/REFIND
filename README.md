# TOTEM
SemEval-2025 Task-3 — Mu-SHROOM

## Installation
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
python -m spacy download en_core_web_lg
```

## Retriever Preparation
```bash
sh scripts/download_en_wiki_dump.sh
sh scripts/preprocess_wiki.sh
sh scripts/build_index.sh
```