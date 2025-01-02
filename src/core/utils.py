#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import sentencepiece as spm

# Enable HTTPS context handling for downloading NLTK resources on Windows systems
if os.name == "nt":
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK tokenizer resource
nltk.download("punkt_tab")


def pad_sentences(sentences: List[List[str]], padding_token: str) -> List[List[str]]:
    """
    Pad sentences to match the length of the longest sentence in the batch.

    Args:
        sentences (List[List[str]]): List of tokenized sentences.
        padding_token (str): Padding token.

    Returns:
        List[List[str]]: List of padded sentences.
    """
    # Determine the maximum sentence length in the batch
    max_length = max(len(sentence) for sentence in sentences)

    # Pad each sentence to the maximum length
    padded_sentences = [
        sentence + [padding_token] * (max_length - len(sentence))
        for sentence in sentences
    ]

    return padded_sentences


def load_corpus(
    file_path: str, language: str, vocab_size: int = 2500
) -> List[List[str]]:
    """
    Load and tokenize a text corpus using SentencePiece.

    Args:
        file_path (str): Path to the corpus file.
        language (str): Language of the corpus ("src" or "tgt").
        vocab_size (int): Size of the vocabulary for subword tokenization.

    Returns:
        List[List[str]]: List of tokenized sentences.
    """
    tokenized_data = []
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f"{language}.model")

    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
            tokens = tokenizer.encode_as_pieces(line.strip())
            if language == "tgt":  # Add start and end tokens for target language
                tokens = ["<s>"] + tokens + ["</s>"]
            tokenized_data.append(tokens)

    return tokenized_data


def load_corpus_with_nltk(file_path: str, language: str) -> List[List[str]]:
    """
    Load and tokenize a text corpus using NLTK.

    Args:
        file_path (str): Path to the corpus file.
        language (str): Language of the corpus ("src" or "tgt").

    Returns:
        List[List[str]]: List of tokenized sentences.
    """
    tokenized_data = []

    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
            tokens = nltk.word_tokenize(line.strip())
            if language == "tgt":  # Add start and end tokens for target language
                tokens = ["<s>"] + tokens + ["</s>"]
            tokenized_data.append(tokens)

    return tokenized_data


def generate_batches(
    data: List[Tuple[List[str], List[str]]], batch_size: int, shuffle: bool = False
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Generate batches of tokenized source and target sentences.

    Args:
        data (List[Tuple[List[str], List[str]]]): List of paired source and target sentences.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the dataset.

    Yields:
        Tuple[List[List[str]], List[List[str]]]: Batched source and target sentences.
    """
    num_batches = math.ceil(len(data) / batch_size)
    indices = list(range(len(data)))

    if shuffle:
        np.random.shuffle(indices)

    for batch_idx in range(num_batches):
        batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_data = [data[idx] for idx in batch_indices]

        # Sort sentences in descending order by source sentence length
        batch_data = sorted(batch_data, key=lambda pair: len(pair[0]), reverse=True)
        source_sentences = [pair[0] for pair in batch_data]
        target_sentences = [pair[1] for pair in batch_data]

        yield source_sentences, target_sentences
