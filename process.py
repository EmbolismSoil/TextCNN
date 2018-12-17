#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import argparse
import codecs
import numpy as np
import re
import os

open = codecs.open


def load_noise_char(path, encoding='utf-8'):
    noise_chars = set()
    noise_chars.add(' ')
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            noise_chars.add(line)

    def __filter_noise_char(s):
        chars = []
        for c in s:
            if c not in noise_chars:
                chars.append(c)
        return ''.join(chars)

    return __filter_noise_char


def build_samples_index(pathin, pathout, vocab_path, filter_noise_char, encoding='utf-8', user_dict=None):
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            vocab.append(line.strip())

    vocab = dict([(w, str(idx)) for idx, w in enumerate(vocab)])
    sms_vocab = []
    labels = []
    contents = []
    with open(pathin, 'r', encoding=encoding) as fin:
        for line in fin:
            items = line.strip().split('\t')
            if len(items) == 5:
                label = items[3]
                sms = items[4]
            elif len(items) == 4:
                label = items[2]
                sms = items[3]
            else:
                print(line)
                continue

            sms = filter_noise_char(sms)
            if len(sms) == 0:
                continue

            sms = list(jieba.cut(sms))
            labels.append('0' if label == '3' else label)

            sms = [re.sub('[0-9]+', '<number>', w) for w in sms]
            sms = [vocab[w] if w in vocab else vocab['<unk>'] for w in sms]
            contents.append(sms)
            sms_vocab += sms
            # fout.write('%s\t%s\n' % (label, ' '.join(sms)))
    sms_vocab = list(set(sms_vocab))
    sms_vocab_map = dict([(old_idx, new_idx) for new_idx, old_idx in enumerate(sms_vocab)])

    lines = []
    with open(pathout, 'w+', encoding='utf-8') as fout:
        for label, sms in zip(labels, contents):
            sms = [str(sms_vocab_map[idx]) for idx in sms]
            lines.append('%s\t%s\n' % (label, ' '.join(sms)))
        fout.write(''.join(lines))

    return [int(w) for w in sms_vocab]


def resample(lines, weights):
    res = []
    for line in lines:
        label = int(line.strip().split('\t')[0])
        label = 0 if label == 3 else label
        weight = weights[label]
        res.extend([line]*weight)

    return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-wrods-path', dest='_stopwords_path', type=str, require=True)
    parser.add_argument('--samples-path', dest='_samples_path', type=str, require=True)
    parser.add_argument('--vocab-path', dest='_vocab_path', type=str, require=True)
    parser.add_argument('--index-path', dest='_index_path', type=str, require=True)
    parser.add_argument('--vecs-path', dest='_vecs_path', type=str, require=True)
    parser.add_argument('--encoding', dest='_encoding', type=str, require=False)
    parser.add_argument('--sms-vecs-path', dest='_sms_vecs_path', type=str, require=True)
    parser.add_argument('--sms-vocab-path', dest='_sms_vocab_path', type=str, require=True)
    parser.add_argument('--train-samples-path', dest='_train_samples_path', type=str, require=True)
    parser.add_argument('--test-samples-path', dest='_test_samples_path', type=str, require=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    filter_noise_char = load_noise_char(args._stopwords_path)
    encoding = 'utf-8' if args._encoding is None else args._encoding
    sms_vocab = build_samples_index(args._samples_path, args._index_path,
                                    args._vocab_path, filter_noise_char,
                                    encoding=encoding)

    with open(args._vocab_path, 'r', encoding=encoding) as fin:
        vocab = [line.strip() for line in fin]

    vecs = np.load(args._vecs_path)
    vocab = [vocab[i] for i in sms_vocab]
    vecs = [vecs[i] for i in sms_vocab]

    np.save(args._sms_vecs_path, vecs)

    with open(args._sms_vocab_path, 'w+', encoding='utf-8') as fout:
        lines = ['%s\n' % w for w in vocab]
        lines = ''.join(lines)
        fout.write(lines)

    with open(args._index_path, 'r', encoding='utf-8') as fin,\
            open(args._train_samples_path, 'w+', encoding='utf-8') as ftrain,\
            open(args._test_samples_path, 'w+', encoding='utf-8') as ftest:
        train_lines = []
        test_lines = []
        for line in fin:
            if np.random.rand() <= 0.8:
                train_lines.append(line)
            else:
                test_lines.append(line)
        ftrain.write(''.join(train_lines))
        ftest.write(''.join(test_lines))

    train_samples_name, ext_name = os.path.splitext(args._train_samples_path)
    train_resample_path = '%s_resample.%s' % (train_samples_name, ext_name)

    with open(args._train_samples_path, 'r', encoding='utf-8') as fin:
        res = resample(fin, weights=[1, 2, 3])

    with open(train_resample_path, 'w+', encoding='utf-8') as fout:
        fout.write(''.join(res))
