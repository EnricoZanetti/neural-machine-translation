#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


class WordEmbeddingLayer(nn.Module):
    """
    A neural network module for converting input words into embeddings for source and target languages.
    """

    def __init__(self, embedding_dim, vocabulary):
        """
        Initialize the embedding layers for source and target languages.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings.
            vocabulary (Vocab): Vocabulary object containing source and target language vocabularies.
        """
        super(WordEmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim

        # Initialize embedding layers for source and target languages
        source_padding_idx = vocabulary.src["<pad>"]
        target_padding_idx = vocabulary.tgt["<pad>"]

        # Embedding layers for source and target languages
        self.source_embedding = nn.Embedding(
            len(vocabulary.src), embedding_dim, source_padding_idx
        )
        self.target_embedding = nn.Embedding(
            len(vocabulary.tgt), embedding_dim, target_padding_idx
        )
