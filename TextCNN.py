import tensorflow as tf
from tensorflow.contrib import slim
from multiprocessing import cpu_count
import numpy as np


class TextCNN(object):
    def __init__(self, sentence_len, wv_mat, word_vec_dim=300,
                 n_class=2, batch_size=50, logdir='./pic',
                 epochs=2, io_buffer_size=1000, padding_value=0,
                 filter_sizes=(3, 4, 5), filter_depth=100, drop_prob=0.5,
                 average_rate=0.95, learning_rate=0.03, activation_fuc='relu',
                 regularizer='l2', regularization_rate=0.001, wv_trainable=False):

        self._wv_trainable = wv_trainable
        self._log_dir = logdir
        self.__build_activation_fuc_map()
        self.__build_regularizer_map()
        self._padding_value = padding_value
        if activation_fuc not in self._activation_fuc_map:
            raise ValueError('activation_fuc : [relu|sigmod|tanh]')
        if regularizer not in self._regularizer_map:
            raise ValueError('regularizer : [l1|l2]')

        self._cpu_count = cpu_count()
        self._io_buffer_size = io_buffer_size

        self._activation_fuc = self._activation_fuc_map[activation_fuc]
        self._regularizer = self._regularizer_map[regularizer]
        self._regularization_rate = regularization_rate
        self._sentence_len = sentence_len
        self._word_vec_dim = word_vec_dim
        self._n_class = n_class
        self._batch_size = batch_size
        self._filter_sizes = filter_sizes
        self._filter_depth = filter_depth
        self._drop_prob = drop_prob
        self._average_rate = average_rate
        self._learning_rate = learning_rate
        self._net = None
        # self._stop_words = self.build_stopwords_set(stop_words_path)

        self._epochs = epochs
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self._sess = tf.Session(config=config)
        self._wv_mat = wv_mat

        self.__build_net()

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)

    def make_nonshuffle_dataset(self, data_path, batch_size, sep='\t'):
        dataset = tf.data.TextLineDataset(data_path)

        def _parse_line(line):
            items = tf.string_split([line], delimiter=sep).values
            c, ws = items[0], items[1]
            ws = tf.string_split([ws], delimiter=' ').values
            ws = tf.string_to_number(ws, tf.int32)
            ws = tf.cond(tf.size(ws) > self._sentence_len, lambda: tf.slice(ws, [0], [self._sentence_len]), lambda: ws)
            c = tf.string_to_number(c, tf.int32)
            return c, ws

        dataset = dataset.map(_parse_line)
        padded_shapes = (tf.TensorShape([]), tf.TensorShape([self._sentence_len]))
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes,
                                       padding_values=(0, self._padding_value))
        return dataset

    def make_nonshuffle_predict_dataset(self, data_path, batch_size, sep=' '):
        def _parse_line(line):
            items = tf.string_split([line], delimiter=' ').values
            ws = tf.string_to_number(items, tf.int32)
            ws = tf.cond(tf.size(ws) > self._sentence_len, lambda: tf.slice(ws, [0], [self._sentence_len]), lambda: ws)
            return ws

        dataset = tf.data.TextLineDataset(data_path)
        dataset = dataset.map(_parse_line)
        dataset = dataset.padded_batch(batch_size, padded_shapes=[self._sentence_len],
                                       padding_values=self._padding_value)
        return dataset

    def make_dataset(self, data_path, sep='\t'):
        dataset = self.make_nonshuffle_dataset(data_path, self._batch_size, sep)
        return dataset.shuffle(self._io_buffer_size)

    def __batch_acc(self):
        predict = tf.argmax(self._net, axis=1)
        acc = tf.equal(predict, self.y_input)
        acc = tf.reduce_mean(tf.cast(acc, tf.float64))
        tf.summary.scalar('acc', acc)
        return acc

    def fit(self, filepath, sep='\t'):
        if not self._wv_trainable:
            x_emb = getattr(self, '_x_emb')

        global_steps = tf.Variable(
            dtype=tf.int32, initial_value=0, trainable=False)
        losses = self.__get_loss()
        self._losses = losses
        self._acc = self.__batch_acc()
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(losses, global_step=global_steps)
        dataset = self.make_dataset(filepath, sep='\t')
        iterator = dataset.make_initializable_iterator()
        x_iter_op, y_iter_op = iterator.get_next()

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self._log_dir, self._sess.graph)
        self._sess.run(tf.initialize_all_variables())

        dp = getattr(self, '_dp')
        steps = 0
        for i in range(self._epochs):
            self._sess.run(iterator.initializer)
            while True:
                try:
                    cls_batch, words_batch = self._sess.run([x_iter_op, y_iter_op])
                    if not self._wv_trainable:
                        feed = {self.x_input: words_batch, self.y_input: cls_batch,
                                x_emb: self._wv_mat, dp: self._drop_prob}
                    else:
                        feed = {self.x_input: words_batch, self.y_input: cls_batch, dp: self._drop_prob}
                except tf.errors.OutOfRangeError:
                    break

                loss, acc, result, _ = self._sess.run([self._losses, self._acc, merged, train_op], feed_dict=feed)
                writer.add_summary(result, steps)
                # loss, acc, _ = self._sess.run([self._losses, self._acc, train_op], feed_dict=feed)
                if steps % 1000 == 0:
                    yield steps, loss, acc
                print('After %d step(s), losses in batch: %g, acc in batch: %g' % (steps, loss, acc))
                steps += 1

    def save(self, path, steps=None):
        saver = tf.train.Saver()
        saver.save(self._sess, path, steps)

    def load(self, checkpoint_dir):
        saver = tf.train.Saver()
        saver.restore(self._sess, tf.train.latest_checkpoint(checkpoint_dir))

    def predict(self, x):
        if not hasattr(self, '_predict'):
            predict = tf.argmax(self._net, axis=1)
            setattr(self, '_predict', predict)

        predict = getattr(self, '_predict')
        dp = getattr(self, '_dp')
        if not self._wv_trainable:
            x_emb = getattr(self, '_x_emb')
            return self._sess.run(predict, feed_dict={self.x_input: x, x_emb: self._wv_mat, dp: self._drop_prob})
        else:
            return self._sess.run(predict, feed_dict={self.x_input: x, dp: self._drop_prob})

    def predict_prob(self, x):
        if not hasattr(self, '_predict_prob'):
            predict_prob = tf.nn.softmax(self._net)
            setattr(self, '_predict_prob', predict_prob)

        predict_prob = getattr(self, '_predict_prob')
        dp = getattr(self, '_dp')
        if not self._wv_trainable:
            x_emb = getattr(self, '_x_emb')
            return self._sess.run(predict_prob, feed_dict={self.x_input: x, x_emb: self._wv_mat, dp: self._drop_prob})
        else:
            return self._sess.run(predict_prob, feed_dict={self.x_input: x, dp: self._drop_prob})

    def __build_net(self):
        x, y_ = self.__embedding_layer()  # input
        self.x_input = x
        self.y_input = y_

        net = getattr(self, 'x_emb')
        conv_layers = []
        for idx, filter_size in enumerate(self._filter_sizes):
            conv_layer = self.__add_conv_layer(net, filter_size, 'conv-layer-%d' % idx)
            pool_layer = self.__pooling(conv_layer, filter_size, 'pool-layer-%d' % idx)
            conv_layers.append(pool_layer)

        net = self.__concat(conv_layers, 'concat-layer')
        net = self.__drop_out(net)
        net = self.__softmax_fcl(net, 'fully-connect-layer')

        self._net = net

    def __build_activation_fuc_map(self):
        self._activation_fuc_map = {
            'relu': tf.nn.relu, 'sigmod': tf.sigmoid, 'tanh': tf.nn.tanh}

    def __build_regularizer_map(self):
        self._regularizer_map = {
            'l2': slim.l2_regularizer, 'l1': slim.l1_regularizer}

    def __embedding_layer(self):
        x = tf.placeholder(tf.int32, shape=[None, self._sentence_len], name='embedding-layer-x')
        y_ = tf.placeholder(tf.int64, shape=[None], name='embedding-layer-y')
        cur_batch_size = tf.shape(x)[0]
        if self._wv_trainable:
            x_emb = tf.Variable(dtype=tf.float64, initial_value=self._wv_mat,
                                name='x_emb', trainable=self._wv_trainable)
        else:
            x_emb = tf.placeholder(tf.float64, shape=[None, self._word_vec_dim], name='x_emb')
            setattr(self, '_x_emb', x_emb)
        x_emb = tf.nn.embedding_lookup(x_emb, x)
        x_emb = tf.reshape(x_emb, shape=[cur_batch_size, self._sentence_len, self._word_vec_dim, 1])
        setattr(self, 'x_emb', x_emb)
        return x, y_

    def __add_conv_layer(self, net, filter_size, name):
        kernel_shape = [filter_size, self._word_vec_dim]
        net = slim.conv2d(net, self._filter_depth,
                          kernel_shape, padding='VALID', scope=name)

        bias = tf.Variable(initial_value=np.random.random([self._filter_depth]), dtype=tf.float64)
        net = tf.nn.bias_add(net, bias)
        net = self._activation_fuc(net)
        return net

    def __pooling(self, net, filter_size, name):
        with tf.variable_scope(name):
            feature_map_len = self._sentence_len - filter_size + 1
            ksize = [1, feature_map_len, 1, 1]
            net = tf.nn.max_pool(
                net, ksize=ksize, padding='VALID', strides=[1, 1, 1, 1])
            return net

    def __concat(self, net, name):
        concated_net = tf.concat(net, axis=3)
        concated_net = slim.flatten(concated_net)
        return concated_net

    def __drop_out(self, net):
        dp_holder = tf.placeholder(dtype=tf.float64, shape=[])
        setattr(self, '_dp', dp_holder)
        return tf.nn.dropout(net, dp_holder)

    def __softmax_fcl(self, net, name):
        regularizer = self._regularizer(self._regularization_rate)
        net = slim.fully_connected(net, self._n_class,
                                   activation_fn=None,
                                   weights_regularizer=regularizer,
                                   biases_regularizer=regularizer, scope=name)
        return net

    def __get_loss(self):
        losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._net, labels=self.y_input))
        regularization_losses = tf.reduce_mean(slim.losses.get_regularization_losses())
        losses += regularization_losses
        tf.summary.scalar('loss', losses)
        return losses
