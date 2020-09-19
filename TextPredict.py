import tensorflow as tf
import math
import time
from Essentials import Vocab, CategoryDict, TextDataSet, Preprocessor

raw_file_untitled = 'data/raw_untitled.txt'
raw_file = 'data/raw.txt'
seg_raw_file = 'data/raw.seg.txt'
vocab_file = 'data/vocab.txt'
category_file = 'data/category.txt'

def get_default_params():
    return tf.contrib.training.HParams(num_embedding_size=128,
                                       num_timesteps=2000,  # 单个句子的标准长度，据此截断或补齐
                                       num_lstm_nodes=[128, 128],
                                       num_lstm_layers=2,
                                       num_fc_nodes=128,
                                       batch_size=1,
                                       clip_lstm_grads=1.0,  # 控制梯度大小
                                       learning_rate=0.001,
                                       num_word_threshold=3  # 词频最低值
                                       )


def append_label(input, output, label="广告"):
    with open(input, mode='r', encoding='UTF-8') as f1:
        lines = f1.readlines()
    with open(output, mode='w', encoding='UTF-8') as f3:
        for line in lines:
            out = label + '\t' + line
            f3.write(out)


def create_model(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # drop-prob = 1 - keep_prob

    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)

    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable('embedding',
                                     [vocab_size, hps.num_embedding_size],
                                     tf.float32
                                     )
        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)
    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hps.num_lstm_nodes[i])
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
        fc1 = tf.layers.dense(last, hps.num_fc_nodes, activation=tf.nn.relu, name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout, num_classes, name='fc2')

    with tf.name_scope('metrics'):
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)

    return inputs, keep_prob, y_pred


# 预处理
append_label(raw_file_untitled, raw_file)
p = Preprocessor(raw_file, seg_raw_file)
p.generate_seg_file()

tf.logging.set_verbosity(tf.logging.INFO)

hps = get_default_params()
vocab = Vocab(vocab_file, hps.num_word_threshold)
category_vocab = CategoryDict(category_file)
raw_dataset = TextDataSet(seg_raw_file, vocab, category_vocab, hps.num_timesteps, False)

vocab_size = vocab.size()
num_classes = category_vocab.size()

inputs, keep_prob, y_pred = create_model(hps, vocab_size, num_classes)

data_size = raw_dataset.size()

init_op = tf.global_variables_initializer()

test_keep_prob_value = 1.0

saver = tf.train.Saver()

result_list=[]

with tf.Session() as sess:
    saver.restore(sess, "./model/text_classify.ckpt")
    for i in range(data_size):
        batch_inputs, batch_labels = raw_dataset.next_batch(hps.batch_size, False)
        outputs_val = int(sess.run(y_pred,
                                   feed_dict={inputs: batch_inputs,
                                              keep_prob: test_keep_prob_value}))
        # print("Predicted value: %s" % category_vocab.id_to_category(outputs_val))
        print("steps: %d,result: %d" % (i+1,outputs_val))  # 0：资讯；1：广告
        result_list.append(outputs_val)

result_file=open("./prediction.txt",mode='w', encoding='UTF-8')
result_file.write(result_list)
result_file.close()