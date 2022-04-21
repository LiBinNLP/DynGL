# -*- coding: utf-8 -*-

import argparse

from supar import BiaffineSemanticDependencyParser
from supar.cmds.cmd import parse


def main():
    PROJ_BASE_PATH = '/mnt/sda1_hd/atur/libin/projects/parser-main/'

    parser = argparse.ArgumentParser(description='Create Biaffine Semantic Dependency Parser.')
    parser.set_defaults(Parser=BiaffineSemanticDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', default=['tag'], choices=['tag', 'char', 'lemma', 'elmo', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--batch_size', default=5000, type=int, help='max num of batch')
    subparser.add_argument('--train', default=PROJ_BASE_PATH+'data/sdp/PSD/czech/train.cs.psd.conllu', help='path to train file')
    subparser.add_argument('--dev', default=PROJ_BASE_PATH+'data/sdp/PSD/czech/dev.cs.psd.conllu', help='path to dev file')
    subparser.add_argument('--test', default=PROJ_BASE_PATH+'data/sdp/PSD/czech/cs.id.psd.conllu', help='path to test file')
    subparser.add_argument('--test_result', default=PROJ_BASE_PATH+'data/sdp/PSD/czech/pred.cs.id.psd.conllu', help='path to test result file')
    subparser.add_argument('--embed', default=PROJ_BASE_PATH+'data/cc.cs.300.vec', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='UNK', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=300, type=int, help='dimension of embeddings')
    subparser.add_argument('--n-embed-proj', default=125, type=int, help='dimension of projected embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    subparser.add_argument('--prob', default=False, help='whether to output probs')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default=PROJ_BASE_PATH+'data/sdp/PSD/czech/cs.ood.psd.conllu', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default=PROJ_BASE_PATH+'data/sdp/PSD/czech/train.cs.psd.conllu', help='path to dataset')
    subparser.add_argument('--pred', default=PROJ_BASE_PATH+'data/sdp/PSD/czech/init.train.cs.psd.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    parse(parser)


if __name__ == "__main__":
    main()
