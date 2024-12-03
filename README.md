# REFIND: Retrieval Augmented Factual Hallucination Detection in Large Language Models
SemEval-2025 Task-3 — Mu-SHROOM

## Installation
```bash
conda create -n REFIND python=3.9
conda activate REFIND
```

```bash
pip install -r requirements.txt
# conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Retriever Preparation
```bash
sh scripts/download_en_wiki_dump.sh
sh scripts/preprocess_wiki.sh
# sh scripts/build_index.sh
```

## Experiment
```bash
sh scripts/run_baseline_random_guess.sh
cat result/scores_baseline_random_guess-0bootstrap.json
cat result/scores_baseline_random_guess-10000bootstrap.json
```

```bash
# WIP
```