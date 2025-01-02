#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .model_embeddings import WordEmbeddingLayer

TranslationHypothesis = namedtuple("TranslationHypothesis", ["tokens", "score"])


class NeuralMachineTranslationModel(nn.Module):
    """
    Neural Machine Translation Model:
    - Bidirectional LSTM Encoder
    - Unidirectional LSTM Decoder
    - Global Attention Mechanism (Luong et al., 2015)
    """

    def __init__(self, embedding_dim, hidden_dim, vocabulary, dropout_prob=0.2):
        """
        Initialize the NMT Model.

        Args:
            embedding_dim (int): Dimensionality of word embeddings.
            hidden_dim (int): Dimensionality of hidden states in LSTM layers.
            vocabulary (Vocab): Vocabulary object containing source and target language vocabularies.
            dropout_prob (float): Dropout rate for regularization.
        """
        super(NeuralMachineTranslationModel, self).__init__()
        self.embedding_layer = WordEmbeddingLayer(embedding_dim, vocabulary)
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.vocabulary = vocabulary

        # Initialize model components
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, bias=True)
        self.decoder = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim, bias=True)
        self.hidden_state_projection = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.cell_state_projection = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.attention_projection = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
        self.target_vocab_projection = nn.Linear(
            hidden_dim, len(vocabulary.tgt), bias=False
        )
        self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(
        self, source_sentences: List[List[str]], target_sentences: List[List[str]]
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of target sentences given source sentences.

        Args:
            source_sentences (List[List[str]]): List of tokenized source sentences.
            target_sentences (List[List[str]]): List of tokenized target sentences with <s> and </s> tokens.

        Returns:
            torch.Tensor: Log-likelihood scores for the batch.
        """
        source_lengths = [len(sentence) for sentence in source_sentences]
        source_tensor = self.vocabulary.src.to_input_tensor(
            source_sentences, device=self.device
        )
        target_tensor = self.vocabulary.tgt.to_input_tensor(
            target_sentences, device=self.device
        )

        encoder_outputs, decoder_init_state = self.encode(source_tensor, source_lengths)
        attention_masks = self.generate_attention_masks(encoder_outputs, source_lengths)
        combined_outputs = self.decode(
            encoder_outputs, attention_masks, decoder_init_state, target_tensor
        )
        log_probs = F.log_softmax(
            self.target_vocab_projection(combined_outputs), dim=-1
        )

        target_masks = (target_tensor != self.vocabulary.tgt["<pad>"]).float()
        gold_log_probs = (
            torch.gather(
                log_probs, index=target_tensor[1:].unsqueeze(-1), dim=-1
            ).squeeze(-1)
            * target_masks[1:]
        )
        batch_scores = gold_log_probs.sum(dim=0)
        return batch_scores

    def encode(
        self, source_tensor: torch.Tensor, source_lengths: List[int]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sentences to obtain encoder hidden states and decoder initial states.

        Args:
            source_tensor (torch.Tensor): Padded source sentences.
            source_lengths (List[int]): Actual lengths of each source sentence.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Encoder hidden states and decoder initial states.
        """
        embedded_sources = self.embedding_layer.source_embedding(source_tensor)
        packed_sources = pack_padded_sequence(
            embedded_sources, source_lengths, enforce_sorted=True
        )
        packed_outputs, (last_hidden, last_cell) = self.encoder(packed_sources)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        decoder_hidden_init = self.hidden_state_projection(
            torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        )
        decoder_cell_init = self.cell_state_projection(
            torch.cat((last_cell[0], last_cell[1]), dim=1)
        )
        return encoder_outputs, (decoder_hidden_init, decoder_cell_init)

    def decode(
        self,
        encoder_outputs: torch.Tensor,
        attention_masks: torch.Tensor,
        decoder_init_state: Tuple[torch.Tensor, torch.Tensor],
        target_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode the encoded representations to generate target sequences.

        Args:
            encoder_outputs (torch.Tensor): Encoder outputs.
            attention_masks (torch.Tensor): Masks for padded encoder outputs.
            decoder_init_state (Tuple[torch.Tensor, torch.Tensor]): Initial hidden and cell states for the decoder.
            target_tensor (torch.Tensor): Padded target sentences.

        Returns:
            torch.Tensor: Combined output tensors for the target sequence.
        """
        target_tensor = target_tensor[:-1]
        decoder_state = decoder_init_state
        previous_output = torch.zeros(
            encoder_outputs.size(0), self.hidden_dim, device=self.device
        )
        combined_outputs = []

        encoder_outputs_proj = self.attention_projection(encoder_outputs)
        target_embeddings = self.embedding_layer.target_embedding(target_tensor)

        for timestep_embedding in torch.split(target_embeddings, 1, dim=0):
            timestep_embedding = timestep_embedding.squeeze(0)
            combined_input = torch.cat((timestep_embedding, previous_output), dim=1)
            decoder_state, combined_output, _ = self.step(
                combined_input,
                decoder_state,
                encoder_outputs,
                encoder_outputs_proj,
                attention_masks,
            )
            combined_outputs.append(combined_output)
            previous_output = combined_output

        return torch.stack(combined_outputs)

    @property
    def device(self) -> torch.device:
        """Returns the device on which the model parameters are located."""
        return self.embedding_layer.source_embedding.weight.device

    @staticmethod
    def load(model_path: str):
        """Load a saved model from file."""
        params = torch.load(model_path, map_location=torch.device("cpu"))
        model = NeuralMachineTranslationModel(**params["model_args"])
        model.load_state_dict(params["state_dict"])
        return model

    def save(self, file_path: str):
        """Save the model to a specified file."""
        params = {
            "model_args": {
                "embedding_dim": self.embedding_layer.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "dropout_prob": self.dropout_prob,
                "vocabulary": self.vocabulary,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(params, file_path)
