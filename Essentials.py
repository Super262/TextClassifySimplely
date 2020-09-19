import jieba_fast as jieba
import tensorflow as tf
import numpy as np
import math


class Preprocessor:
    def __init__(self, input_file, output_seg_file):
        self._input_file = input_file
        self._output_seg_file = output_seg_file

    def generate_seg_file(self):
        # 分割输入文件中每行的句子
        with open(self._input_file, mode='r', encoding='UTF-8') as f1:
            lines = f1.readlines()
        with open(self._output_seg_file, mode='w', encoding='UTF-8') as f2:
            for line in lines:
                label, content = line.strip('\r\n').split('\t', 1)  # 先去掉换行符和回车符，再按制表符拆分
                word_iter = jieba.cut(content)
                word_content = ''
                for word in word_iter:
                    word = word.strip(' ')  # 去掉空格；处理后的word可能为空字符串
                    if word != '':
                        word_content += word + ' '
                out_line = '%s\t%s\n' % (label, word_content.strip(' '))
                f2.write(out_line)

    def generate_vocab_file(self, output_vocab_file):
        with open(self._output_seg_file, mode='r', encoding='UTF-8') as f1:
            lines = f1.readlines()
        word_dict = {}  # 保存词频，以便筛选
        for line in lines:
            label, content = line.strip('\r\n').split('\t', 1)
            # 统计词频
            for word in content.strip('\r\n').split():  # 按空格拆分
                word_dict.setdefault(word, 0);
                word_dict[word] += 1

        # 排序；得到一个列表，形如[(word, frequency), ...]
        sorted_word_dict = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
        with open(output_vocab_file, mode='w', encoding='UTF-8') as f2:
            f2.write('<UNK>\t1000000\n')
            for item in sorted_word_dict:
                f2.write('%s\t%d\n' % (item[0], item[1]))

    def generate_catagory_dict(self, category_file):
        with open(self._input_file, mode='r', encoding='UTF-8') as f1:
            lines = f1.readlines()
        category_dict = {}  # 统计每个类别出现的频率
        for line in lines:
            label, content = line.strip('\r\n').split('\t', 1)
            category_dict.setdefault(label, 0)
            category_dict[label] += 1
        with open(category_file, mode='w', encoding='UTF-8') as f2:
            for category in category_dict:
                line = '%s\n' % category  # 默认以行号作为id
                f2.write(line)


class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename, mode='r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.strip('\r\n').split('\t', 1)
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                self._unk = idx
            self._word_to_id[word] = idx

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return word_ids


class CategoryDict:
    def __init__(self, filename):
        self._category_to_id = {}
        self._id_to_category = {}
        with open(filename, mode='r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            category = line.strip('\r\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx
            self._id_to_category[idx] = category

    def category_to_id(self, category):
        if not category in self._category_to_id:
            raise Exception("%s is not in our category list" % category)
        return self._category_to_id[category]

    def id_to_category(self, id):
        if not id in self._id_to_category:
            raise Exception("%s is not in our id list" % id)
        return self._id_to_category[id]

    def size(self):
        return len(self._category_to_id)


class TextDataSet:
    def __init__(self, filename, vocab, category_vocab, num_timesteps, random):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps
        self._inputs = []  # 矩阵
        self._labels = []  # 向量
        self._indicator = 0
        self._random = random
        self._prase_file(filename)

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._labels = self._labels[p]

    def _prase_file(self, filename):
        tf.logging.info('Loading data from %s' % filename)
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            label, content = line.strip('\r\n').split('\t', 1)
            id_label = self._category_vocab.category_to_id(label)
            id_words = self._vocab.sentence_to_id(content)

            if (len(id_words) >= self._num_timesteps + 1):
                id_words = id_words[0: self._num_timesteps]
            else:
                padding_num = self._num_timesteps - len(id_words)
                id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._labels.append(id_label)

        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._labels = np.asarray(self._labels, dtype=np.int32)
        if self._random:
            self._random_shuffle()

    def next_batch(self, batch_size, shuffle):
        end_indicator = self._indicator + batch_size
        if end_indicator >= len(self._inputs):
            if shuffle:
                self._random_shuffle()
                self._indicator = 0
                end_indicator = batch_size

        if end_indicator >= len(self._inputs):
            raise Exception("batch size: %d is too large" % batch_size)

        batch_inputs = self._inputs[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_labels

    def size(self):
        return len(self._inputs)


class TextModel:
    def __init__(self, hps, vocab_size, num_classes):
        self._hps = hps
        self._vocab_size = vocab_size
        self._num_classes = num_classes

    def create_model(self):
        num_timesteps = self._hps.num_timesteps
        batch_size = self._hps.batch_size

        inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
        outputs = tf.placeholder(tf.int32, (batch_size,))
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # drop-prob = 1 - keep_prob

        global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)  # 记录当前训练到哪一步

        embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)

        with tf.variable_scope('embedding', initializer=embedding_initializer):
            embeddings = tf.get_variable('embedding',
                                         [self._vocab_size, self._hps.num_embedding_size],
                                         tf.float32
                                         )
            # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
            embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)
        scale = 1.0 / math.sqrt(self._hps.num_embedding_size + self._hps.num_lstm_nodes[-1]) / 3.0
        lstm_init = tf.random_uniform_initializer(-scale, scale)
        with tf.variable_scope('lstm_nn', initializer=lstm_init):
            cells = []
            for i in range(self._hps.num_lstm_layers):
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self._hps.num_lstm_nodes[i])
                # cell = tf.contrib.rnn.BasicLSTMCell(self._hps.num_lstm_nodes[i], state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            initial_state = cell.zero_state(batch_size, tf.float32)

            # rnn_outputs是三维矩阵，[batch_size, num_timesteps, lstm_outputs[-1]]
            # _ 保存隐含状态，这里不会用到
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, embed_inputs, initial_state=initial_state, swap_memory=False,
                                               parallel_iterations=2304)
            last = rnn_outputs[:, -1, :]

        fc_init = tf.initializers.variance_scaling(1.0)
        with tf.variable_scope('fc', initializer=fc_init):
            fc1 = tf.layers.dense(last, self._hps.num_fc_nodes, activation=tf.nn.relu, name='fc1')
            fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
            logits = tf.layers.dense(fc1_dropout, self._num_classes, name='fc2')

        with tf.name_scope('metrics'):
            softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
            loss = tf.reduce_mean(softmax_loss)
            y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)
            correct_pred = tf.equal(outputs, y_pred)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope('train_op'):
            tvars = tf.trainable_variables()
            for var in tvars:
                tf.logging.info('variable name: %s' % var.name)
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self._hps.clip_lstm_grads)
            optimizer = tf.train.AdamOptimizer(self._hps.learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        return (inputs, outputs, keep_prob), (loss, accuracy), (train_op, global_step)
