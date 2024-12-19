# REFIND: Retrieval Augmented Factual Hallucination Detection in Large Language Models

<div align="center">
    <img src="https://github.com/user-attachments/assets/2fbf29ca-560f-4924-b020-6f1b3488dac6" alt="Overview of REFIND" width="75%">
</div>
<br>

## Task & Dataset Info
[SemEval-2025 Task-3 — Mu-SHROOM](https://helsinki-nlp.github.io/shroom/)


## Usage
### Installation
```bash
conda create -n REFIND python=3.9
conda activate REFIND
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
```

### Preparation
Download Mu-SHROOM Dataset from [Official Website](https://helsinki-nlp.github.io/shroom/#data) and put it in the [`data` directory](data/README.md).

```bash
# Retriever Preprocessing
sh scripts/preprocess_wiki.sh
```

### Experiment
```bash
# Our Method
sh scripts/run_REFIND.sh

# Baselines
sh scripts/run_random_guess.sh
sh scripts/run_XLM-R.sh
sh scripts/run_FAVA.sh

## Evaluation
sh scripts/evaluate.sh
```

## References
- [Official Website of Mu-SHROOM](https://helsinki-nlp.github.io/shroom/)
- [Official GitHub Repository of Mu-SHROOM](https://github.com/Helsinki-NLP/shroom)
- [Official Participant Kit of Mu-SHROOM](https://a3s.fi/mickusti-2007780-pub/participant_kit.zip)