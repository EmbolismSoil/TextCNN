#!/usr/bin/env python
import os
import argparse
import codecs
import jieba
import re
from datetime import datetime, timedelta
import gzip
from TextCNN import TextCNN
import tensorflow as tf
import numpy as np
import shutil
import io

SMS_ITEMS_LEN = 5
SMS_INDEX = 4
BIG_BUFFER_LINES = 100000
MID_BUFFER_LINES = 50000
SMALL_BUFFER_LINES = 20000


def load_noise_char(path, encoding='utf-8'):
    noise_chars = set()
    noise_chars.add(' ')
    with codecs.open(path, 'r', encoding=encoding, buffering=BIG_BUFFER_LINES) as f:
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


def unzip(gz_path, out_path):
    with gzip.open(gz_path, 'rb') as fin,\
            codecs.open(out_path, 'wb+', buffering=BIG_BUFFER_LINES) as fout:
        fin = io.BufferedReader(fin, buffer_size=BIG_BUFFER_LINES)
        shutil.copyfileobj(fin, fout)
        fout.flush()


def merge_result(result_path, samples_path, words_path, out_path):
    with codecs.open(result_path, 'r', encoding='utf-8', buffering=MID_BUFFER_LINES) as fr, \
            codecs.open(samples_path, 'rb', buffering=MID_BUFFER_LINES) as fs,\
            codecs.open(words_path, 'rb', buffering=MID_BUFFER_LINES) as fw,\
            codecs.open(out_path, 'w+', encoding='utf-8', buffering=MID_BUFFER_LINES) as fout:
        for lr, ls, lw in zip(fr, fs, fw):
            lr = lr.strip()
            try:
                ls = ls.decode('utf-8')
                lw = lw.decode('utf-8')
            except UnicodeDecodeError:
                continue
            ls = ls.strip()
            lw = lw.strip()
            line = '%s\t%s\t%s\n' % (lr, ls, lw)
            fout.write(line)

        fout.flush()


def to_index_file(samples_path, index_path, words_path, filter_noise_char, vocab):
    # index_lines = []
    # words_lines = []
    with codecs.open(samples_path, 'rb', buffering=MID_BUFFER_LINES) as fin,\
            codecs.open(index_path, 'w+', encoding='utf-8', buffering=MID_BUFFER_LINES) as fiout,\
            codecs.open(words_path, 'w+', encoding='utf-8', buffering=MID_BUFFER_LINES) as fwout:
        while True:
            try:
                for line in fin:
                    line = line.decode('utf-8')
                    items = line.strip().split('\t')
                    if len(items) == SMS_ITEMS_LEN:
                        sms = items[SMS_INDEX]
                    else:
                        fiout.write('%s\n' % vocab['<unk>'])
                        fwout.write('%s\n' % '<unk>')
                        continue

                    sms = filter_noise_char(sms)
                    if len(sms) == 0:
                        fiout.write('%s\n' % vocab['<unk>'])
                        fwout.write('%s\n' % '<unk>')
                        continue

                    sms = jieba.cut(sms)
                    sms = [re.sub('[0-9]+', '<number>', w) for w in sms]
                    fwout.write('%s\n' % ' '.join(sms))
                    sms = [vocab[w] if w in vocab else vocab['<unk>'] for w in sms]
                    fiout.write('%s\n' % ' '.join(sms))
            except UnicodeDecodeError:
                fiout.write('%s\n' % vocab['<unk>'])
                continue
            break

        fiout.flush()
        fwout.flush()

    # with codecs.open(index_path, 'w+', encoding='utf-8') as fout:
        # fout.write(''.join(index_lines))

    # with codecs.open(words_path, 'w+', encoding='utf-8') as fout:
        # fout.write(''.join(words_lines))


def predict(args, text_cnn):
    samples_path = args.samples_path
    vocab_path = args.vocab_path
    result_path = args.result_path

    noise_char_path = args.noise_char_path
    date = args.date
    if date is None:
        # date = datetime.now().strftime('%Y%m%d')
        date = datetime.now() - timedelta(days=1)
        date = date.strftime('%Y%m%d')

    filter_noise_char = load_noise_char(noise_char_path)

    with codecs.open(vocab_path, 'r', encoding='utf-8', buffering=BIG_BUFFER_LINES) as fin:
        vocab = [line.strip() for line in fin]

    vocab = dict([(w, str(idx)) for idx, w in enumerate(vocab)])

    zip_samples_path = samples_path + '/%s_samples.gz' % date
    unzip_samples_path = samples_path + '/%s_samples.csv' % date

    if not os.path.isfile(zip_samples_path):
        return

    unzip(zip_samples_path, unzip_samples_path)
    samples_path = unzip_samples_path

    samples_name, samples_ext = os.path.splitext(samples_path)
    index_path = samples_name + '.index'
    words_path = samples_name + '.words'

    # clean_file(samples_path)
    to_index_file(samples_path, index_path, words_path, filter_noise_char, vocab)

    dataset = text_cnn.make_nonshuffle_predict_dataset(index_path, 1000)

    iterator = dataset.make_initializable_iterator()
    sms_it_op = iterator.get_next()
    text_cnn.run(iterator.initializer)

    result_path = result_path + '/%s_result.csv' % date
    with codecs.open(result_path, 'w+', encoding='utf-8', buffering=BIG_BUFFER_LINES) as fout:
        while True:
            try:
                sms_batch = text_cnn.run(sms_it_op)
            except tf.errors.OutOfRangeError:
                break

            y_prob = text_cnn.predict_prob(sms_batch)
            content = ''.join(map(lambda x: '%.9f\t%.9f\t%.9f\n' % tuple(x), y_prob))
            fout.write(content)
        fout.flush()

    result_name, result_ext = os.path.splitext(result_path)
    merged_result_path = result_name + '.merged'
    merge_result(result_path, samples_path, words_path, merged_result_path)


def date_iter(start, end, f='%Y%m%d'):
    start_date = datetime.strptime(start, f)
    end_date = datetime.strptime(end, f)
    while start_date <= end_date:
        yield start_date.strftime(f)
        start_date = start_date + timedelta(days=1)


def get_args():
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='command')
    sub_parser.required = True
    train_parser = sub_parser.add_parser('train', help='train textcnn model')

    train_parser.add_argument('--log-dir', dest='logdir', type=str, required=True)
    train_parser.add_argument('--wordvec-path', dest='wordvec_path', type=str, required=True)
    train_parser.add_argument('--samples-path', dest='samples_path', type=str, required=True)
    train_parser.add_argument('--model-path', dest='model_path', type=str, required=True)

    test_parser = sub_parser.add_parser('test', help='test text model')
    test_parser.add_argument('--model-path', dest='model_path', type=str, required=True)
    test_parser.add_argument('--samples-path', dest='samples_path', type=str, required=True)
    test_parser.add_argument('--vocab-path', dest='vocab_path', type=str, required=True)
    test_parser.add_argument('--result-path', dest='result_path', type=str, required=True)
    test_parser.add_argument('--wordvec-path', dest='wordvec_path', type=str, required=True)

    test_parser = sub_parser.add_parser('predict', help='predict')
    test_parser.add_argument('--model-path', dest='model_path', type=str, required=True)
    test_parser.add_argument('--samples-path', dest='samples_path', type=str, required=True)
    test_parser.add_argument('--vocab-path', dest='vocab_path', type=str, required=True)
    test_parser.add_argument('--result-path', dest='result_path', type=str, required=True)
    test_parser.add_argument('--wordvec-path', dest='wordvec_path', type=str, required=True)
    test_parser.add_argument('--noise-char-path', dest='noise_char_path', type=str, required=True)
    test_parser.add_argument('--date', dest='date', type=str, required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.command == 'train':
        logdir = args.logdir
        wordvec_path = args.wordvec_path
        samples_path = args.samples_path
        model_path = args.model_path

        wordvec = np.load(wordvec_path)

        text_cnn = TextCNN(80, wordvec, 200, learning_rate=0.001, epochs=50, logdir=logdir, io_buffer_size=30000,
                           n_class=3, batch_size=100, filter_sizes=(2, 3, 4), filter_depth=100, padding_value=4072)

        for steps, loss, acc in text_cnn.fit(samples_path):
            print('save checkpoint!')
            text_cnn.save(model_path, steps)
        text_cnn.save(model_path)

    elif args.command == 'predict':
        wordvec_path = args.wordvec_path
        wordvec = np.load(wordvec_path)
        start_date = datetime.strptime('20180601', '%Y%m%d')
        text_cnn = TextCNN(80, wordvec, 200, learning_rate=0.001, epochs=50,
                           logdir=None, io_buffer_size=30000, drop_prob=1.0,
                           n_class=3, batch_size=100, filter_sizes=(2, 3, 4),
                           filter_depth=100, padding_value=4072)

        model_path = args.model_path
        text_cnn.load(model_path)
        if args.date is None:
            predict(args, text_cnn)
        else:
            dates = args.date.split('-')
            if len(dates) == 2:
                for d in date_iter(dates[0], dates[1]):
                    args.date = d
                    predict(args, text_cnn)
            else:
                predict(args, text_cnn)

    else:
        from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

        model_path = args.model_path
        samples_path = args.samples_path
        vocab_path = args.vocab_path
        result_path = args.result_path
        wordvec_path = args.wordvec_path
        wordvec = np.load(wordvec_path)

        text_cnn = TextCNN(80, wordvec, 200, learning_rate=0.001, epochs=50, logdir=None, io_buffer_size=30000,
                           n_class=3, batch_size=100, filter_sizes=(2, 3, 4), filter_depth=100, padding_value=4072)

        text_cnn.load(model_path)

        labels = []
        y_pre = []

        test_dataset = text_cnn.make_nonshuffle_dataset(samples_path, 1000)
        iterator = test_dataset.make_initializable_iterator()
        labels_it_op, x_it_op = iterator.get_next()
        text_cnn._sess.run(iterator.initializer)

        with codecs.open(vocab_path, 'r', encoding='utf-8') as fin, \
                codecs.open(result_path, 'w+', encoding='utf-8') as fout:

            vocab = []
            for line in fin:
                vocab.append(line.strip())

            while True:
                try:
                    x_batch, labels_batch = text_cnn._sess.run([x_it_op, labels_it_op])
                    labels += list(labels_batch)
                except tf.errors.OutOfRangeError:
                    break

                pre_batch = list(text_cnn.predict(x_batch))
                y_pre += pre_batch

                result_lines = []
                for idx, (r, p) in enumerate(zip(labels_batch, pre_batch)):
                    text = [vocab[i] for i in x_batch[idx]]
                    if r != p:
                        line = '[error]\t%d\t%d\t%s' % (r, p, ' '.join(text))
                    else:
                        line = '[ok]\t%d\t%d\t%s' % (r, p, ' '.join(text))
                    result_lines.append(line)
                fout.write('\n'.join(result_lines))

            y = labels
            cm = confusion_matrix(y, y_pre)
            acc = accuracy_score(y, y_pre)
            f1 = f1_score(y, y_pre, average='macro')
            recall = recall_score(y, y_pre, average='macro')
            fout.write('test result:\nacc: %f, recall: %f, f1: %f\n' % (acc, recall, f1))
            fout.write('confusion matrix: \n%s\n' % str(cm))
