#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Machine Translation Training and Decoding Script.

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               Show this help message.
    --gpu                                   Use GPU for training and decoding.
    --compile                               Compile the model for optimization.
    --no-compile                            Disable model compilation.
    --backend=<str>                         Backend for compilation [default: inductor].
    --train-src=<file>                      Path to training source file.
    --train-tgt=<file>                      Path to training target file.
    --dev-src=<file>                        Path to development source file.
    --dev-tgt=<file>                        Path to development target file.
    --vocab=<file>                          Path to vocabulary file.
    --seed=<int>                            Random seed [default: 0].
    --batch-size=<int>                      Batch size for training [default: 32].
    --embed-size=<int>                      Embedding size [default: 256].
    --hidden-size=<int>                     Hidden size for model layers [default: 256].
    --clip-grad=<float>                     Gradient clipping threshold [default: 5.0].
    --log-every=<int>                       Frequency of logging during training [default: 10].
    --max-epoch=<int>                       Maximum number of epochs [default: 30].
    --input-feed                            Enable input feeding mechanism.
    --patience=<int>                        Number of iterations to wait before reducing learning rate [default: 5].
    --max-num-trial=<int>                   Maximum number of trials before early stopping [default: 5].
    --lr-decay=<float>                      Learning rate decay factor [default: 0.5].
    --beam-size=<int>                       Beam size for decoding [default: 5].
    --sample-size=<int>                     Sample size for evaluation [default: 5].
    --lr=<float>                            Initial learning rate [default: 0.001].
    --uniform-init=<float>                  Range for uniform parameter initialization [default: 0.1].
    --save-to=<file>                        Path to save the trained model [default: model.bin].
    --valid-niter=<int>                     Frequency of validation [default: 2000].
    --dropout=<float>                       Dropout rate [default: 0.3].
    --max-decoding-time-step=<int>          Maximum decoding time steps [default: 70].
"""

import math
import sys
import pickle
import time

from docopt import docopt

# from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import sacrebleu
from core import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from core import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, batch_size=32):
    """Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.0
    cum_tgt_words = 0.0

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents
            )  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(
    references: List[List[str]], hypotheses: List[Hypothesis]
) -> float:
    """Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == "<s>":
        references = [ref[1:-1] for ref in references]

    # detokenize the subword pieces to get full sentences
    detokened_refs = ["".join(pieces).replace("▁", " ") for pieces in references]
    detokened_hyps = ["".join(hyp.value).replace("▁", " ") for hyp in hypotheses]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_hyps, [detokened_refs])

    return bleu.score


def train(args: Dict):
    """Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(
        args["--train-src"], source="src", vocab_size=21000
    )  # EDIT: NEW VOCAB SIZE
    train_data_tgt = read_corpus(args["--train-tgt"], source="tgt", vocab_size=8000)

    dev_data_src = read_corpus(args["--dev-src"], source="src", vocab_size=3000)
    dev_data_tgt = read_corpus(args["--dev-tgt"], source="tgt", vocab_size=2000)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args["--batch-size"])
    clip_grad = float(args["--clip-grad"])
    valid_niter = int(args["--valid-niter"])
    log_every = int(args["--log-every"])
    model_save_path = args["--save-to"]

    vocab = Vocab.load(args["--vocab"])

    # model = NMT(embed_size=int(args['--embed-size']),                                 # EDIT: 4X EMBED AND HIDDEN SIZES
    #             hidden_size=int(args['--hidden-size']),
    #             dropout_rate=float(args['--dropout']),
    #             vocab=vocab)

    model = NMT(
        embed_size=1024,
        hidden_size=1024,
        dropout_rate=float(args["--dropout"]),
        vocab=vocab,
    )

    if args["--compile"] == True:
        try:
            model = torch.compile(model, backend=args["--backend"])
            print(f"NMT model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    model.train()

    uniform_init = float(args["--uniform-init"])
    if np.abs(uniform_init) > 0.0:
        print(
            "uniformly initialize parameters [-%f, +%f]" % (uniform_init, uniform_init),
            file=sys.stderr,
        )
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt["<pad>"]] = 0

    device = setup_device(args["--gpu"])
    print("use device: %s" % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args["--lr"]))
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)                       # EDIT: SMALLER LEARNING RATE

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = (
        report_tgt_words
    ) = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print("begin Maximum Likelihood training")

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(
            train_data, batch_size=train_batch_size, shuffle=True
        ):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents
            )  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print(
                    "epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f "
                    "cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec"
                    % (
                        epoch,
                        train_iter,
                        report_loss / report_examples,
                        math.exp(report_loss / report_tgt_words),
                        cum_examples,
                        report_tgt_words / (time.time() - train_time),
                        time.time() - begin_time,
                    ),
                    file=sys.stderr,
                )

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.0

            # perform validation
            if train_iter % valid_niter == 0:
                print(
                    "epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d"
                    % (
                        epoch,
                        train_iter,
                        cum_loss / cum_examples,
                        np.exp(cum_loss / cum_tgt_words),
                        cum_examples,
                    ),
                    file=sys.stderr,
                )

                cum_loss = cum_examples = cum_tgt_words = 0.0
                valid_num += 1

                print("begin validation ...", file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(
                    model, dev_data, batch_size=128
                )  # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print(
                    "validation: iter %d, dev. ppl %f" % (train_iter, dev_ppl),
                    file=sys.stderr,
                )

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(
                    hist_valid_scores
                )
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print(
                        "save currently the best model to [%s]" % model_save_path,
                        file=sys.stderr,
                    )
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + ".optim")
                elif patience < int(args["--patience"]):
                    patience += 1
                    print("hit patience %d" % patience, file=sys.stderr)

                    if patience == int(args["--patience"]):
                        num_trial += 1
                        print("hit #%d trial" % num_trial, file=sys.stderr)
                        if num_trial == int(args["--max-num-trial"]):
                            print("early stop!", file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]["lr"] * float(args["--lr-decay"])
                        print(
                            "load previously best model and decay learning rate to %f"
                            % lr,
                            file=sys.stderr,
                        )

                        # load model
                        params = torch.load(
                            model_save_path,
                            map_location=lambda storage, loc: storage,
                            weights_only=True,
                        )
                        model.load_state_dict(params["state_dict"])
                        model = model.to(device)

                        print("restore parameters of the optimizers", file=sys.stderr)
                        optimizer.load_state_dict(
                            torch.load(model_save_path + ".optim", weights_only=True)
                        )

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args["--max-epoch"]):
                    print("reached maximum number of epochs!", file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    """Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print(
        "load test source sentences from [{}]".format(args["TEST_SOURCE_FILE"]),
        file=sys.stderr,
    )
    test_data_src = read_corpus(args["TEST_SOURCE_FILE"], source="src", vocab_size=3000)
    if args["TEST_TARGET_FILE"]:
        print(
            "load test target sentences from [{}]".format(args["TEST_TARGET_FILE"]),
            file=sys.stderr,
        )
        test_data_tgt = read_corpus(
            args["TEST_TARGET_FILE"], source="tgt", vocab_size=2000
        )

    print("load model from {}".format(args["MODEL_PATH"]), file=sys.stderr)
    model = NMT.load(args["MODEL_PATH"])
    model = model.to(setup_device(args["--gpu"]))

    hypotheses = beam_search(
        model,
        test_data_src,
        #  beam_size=int(args['--beam-size']),                      EDIT: BEAM SIZE USED TO BE 5
        beam_size=10,
        max_decoding_time_step=int(args["--max-decoding-time-step"]),
    )

    if args["TEST_TARGET_FILE"]:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print("Corpus BLEU: {}".format(bleu_score), file=sys.stderr)

    with open(args["OUTPUT_FILE"], "w", encoding="utf-8") as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = "".join(top_hyp.value).replace("▁", " ")
            f.write(hyp_sent + "\n")


def beam_search(
    model: NMT,
    test_data_src: List[List[str]],
    beam_size: int,
    max_decoding_time_step: int,
) -> List[List[Hypothesis]]:
    """Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc="Decoding", file=sys.stdout):
            example_hyps = model.beam_search(
                src_sent,
                beam_size=beam_size,
                max_decoding_time_step=max_decoding_time_step,
            )

            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


def setup_device(gpu: bool):
    """Setup the device used by PyTorch."""

    device = torch.device("cpu")

    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")

    return device


def main():
    """Main func."""
    args = docopt(__doc__)
    print(args)

    # Check pytorch version
    assert (
        torch.__version__ >= "1.0.0"
    ), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(
        torch.__version__
    )

    # seed the random number generators
    seed = int(args["--seed"])
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args["train"]:
        train(args)
    elif args["decode"]:
        decode(args)
    else:
        raise RuntimeError("invalid run mode")


if __name__ == "__main__":
    main()
