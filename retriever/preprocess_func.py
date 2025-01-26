from typing import List


# English (EN)
from langchain_community.retrievers.bm25 import (
    default_preprocessing_func as EN_preprocessing_func,
)


# Arabic (AR)
from pyarabic.araby import tokenize


def AR_preprocessing_func(text: str) -> List[str]:
    return tokenize(text)


# German (DE)
from nltk.tokenize import word_tokenize


def DE_preprocessing_func(text: str) -> List[str]:
    return word_tokenize(text, language="german")


# Spanish (ES)
def ES_preprocessing_func(text: str) -> List[str]:
    return word_tokenize(text, language="spanish")


# Finnish (FI)
def FI_preprocessing_func(text: str) -> List[str]:
    return word_tokenize(text, language="finnish")


# French (FR)
def FR_preprocessing_func(text: str) -> List[str]:
    return word_tokenize(text, language="french")


# Italian (IT)
def IT_preprocessing_func(text: str) -> List[str]:
    return word_tokenize(text, language="italian")


# Hindi (HI)
from indicnlp.tokenize import indic_tokenize


def HI_preprocessing_func(text: str) -> List[str]:
    return indic_tokenize.trivial_tokenize(text, lang="hi")


# Swedish (SV)
def SV_preprocessing_func(text: str) -> List[str]:
    return word_tokenize(text, language="swedish")


# Chinese (ZH)
from jieba import cut


def ZH_preprocessing_func(text: str) -> List[str]:
    return list(cut(text))


# Basque (EU)
import re


def EU_preprocessing_func(text: str) -> List[str]:
    # Basic tokenization based on whitespace and punctuation
    return re.findall(r"\b\w+\b", text)


# Catalan (CA)
def CA_preprocessing_func(text: str) -> List[str]:
    return text.split()


# Czech (CS)
def CS_preprocessing_func(text: str) -> List[str]:
    return word_tokenize(text, language="czech")