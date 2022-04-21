# -*- coding: utf-8 -*-

import argparse
from supar.cmds.cmd import parse
from supar.parsers.sdp import DyGLSemanticDependencyParser


def main():
    PROJ_BASE_PATH = '/mnt/sda1_hd/atur/libin/projects/DyGLSDP/'
    EMBEDDING_PATH = '/mnt/sda1_hd/atur/libin/projects/wordvec/'
    BERT_PATH = '/mnt/sda1_hd/atur/libin/projects/BERT/'

    mode = 'trial' # trial/formal
    language = 'english'
    formalism = 'DM'

    abbreviation = {
        'english': 'en',
        'chinese': 'cz',
        'czech': 'cs',
        'DM': 'dm',
        'PAS': 'pas',
        'PSD': 'psd'
    }

    embedding_path = {
        'english': '{}/glove.6B.100d.txt'.format(language),
        'chinese': '{}/cc.zh.300.vec'.format(language),
        'czech': '{}/cc.cs.300.vec'.format(language),
    }

    unk_token = {
        'english': 'unk',
        'chinese': 'UNK',
        'czech': 'UNK',
    }

    bert_path = {
        'english': 'bert-base-cased',
        'chinese': BERT_PATH + 'bert_base_chinese_pytorch/',
        'czech': BERT_PATH + 'bg_cs_pl_ru_cased_L-12_H-768_A-12_pt/'
    }

    data_path = {
        'trial': {
            'train': 'data/sdp/trial/{}/{}.conllu'.format(formalism, abbreviation[formalism]),
            'dev': 'data/sdp/trial/{}/{}.conllu'.format(formalism, abbreviation[formalism]),
            'test': 'data/sdp/trial/{}/{}.conllu'.format(formalism, abbreviation[formalism]),
            'eval': 'data/sdp/trial/{}/{}.conllu'.format(formalism, abbreviation[formalism]),
            'test_result': 'data/sdp/trial/{}/pred.{}.conllu'.format(formalism, abbreviation[formalism]),
        },
        'formal': {
            'train': 'data/sdp/{}/{}/train.{}.{}.conllu'.format(formalism, language, abbreviation[language], abbreviation[formalism]),
            'dev': 'data/sdp/{}/{}/dev.{}.{}.conllu'.format(formalism, language, abbreviation[language], abbreviation[formalism]),
            'test': 'data/sdp/{}/{}/{}.id.{}.conllu'.format(formalism, language, abbreviation[language], abbreviation[formalism]),
            'eval': 'data/sdp/{}/{}/{}.ood.{}.conllu'.format(formalism, language, abbreviation[language], abbreviation[formalism]),
            'test_result': 'data/sdp/{}/{}/pred.{}.id.{}.conllu'.format(formalism, language, abbreviation[language], abbreviation[formalism])
        }
    }

    train = data_path[mode]['train']
    dev = data_path[mode]['dev']
    test = data_path[mode]['test']
    eval = data_path[mode]['eval']
    test_result = data_path[mode]['test_result']

    parser = argparse.ArgumentParser(description='Create Graph Structure Learning Semantic Dependency Parser.')
    parser.set_defaults(Parser=DyGLSemanticDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    # subparser.add_argument('--feat', '-f', default=['tag', 'char', 'lemma', 'bert'], choices=['tag', 'char', 'lemma', 'elmo', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--batch_size', default=2000, type=int, help='max num of batch')
    subparser.add_argument('--train', default=PROJ_BASE_PATH + train, help='path to train file')
    subparser.add_argument('--dev', default=PROJ_BASE_PATH + dev, help='path to dev file')
    subparser.add_argument('--test', default=PROJ_BASE_PATH + test, help='path to test file')
    subparser.add_argument('--test_result', default=PROJ_BASE_PATH + test_result, help='path to test result file')
    subparser.add_argument('--embed', default=EMBEDDING_PATH + embedding_path[language], help='path to pretrained embeddings')
    subparser.add_argument('--unk', default=unk_token[language], help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--n-embed-proj', default=125, type=int, help='dimension of projected embeddings')
    subparser.add_argument('--bert', default=bert_path[language], help='which BERT model to use')
    subparser.add_argument('--prob', default=False, help='whether to output probs')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default=PROJ_BASE_PATH + eval, help='path to dataset')
    subparser.add_argument('--batch_size', default=2000, type=int, help='max num of batch')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default=PROJ_BASE_PATH+'data/sdp/DM/english/train.en.dm.conllu', help='path to dataset')
    subparser.add_argument('--pred', default=PROJ_BASE_PATH+'data/sdp/DM/english/pred.train.en.dm.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    parse(parser)


if __name__ == "__main__":
    main()
