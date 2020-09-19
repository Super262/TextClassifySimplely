import tensorflow as tf
import time
from Essentials import Vocab, CategoryDict, TextDataSet, TextModel, Preprocessor

train_file = './data/train.txt'
seg_train_file = 'data/train.seg.txt'
vocab_file = 'data/vocab.txt'
category_file = 'data/category.txt'

def get_default_params():
    return tf.contrib.training.HParams(num_embedding_size=64,
                                       num_timesteps=2000,  # 单个句子的标准长度，据此截断或补齐
                                       num_lstm_nodes=[64, 64],
                                       num_lstm_layers=2,
                                       num_fc_nodes=64,
                                       batch_size=50,
                                       clip_lstm_grads=1.0,  # 控制梯度大小
                                       learning_rate=0.001,
                                       num_word_threshold=3  # 词频最低值
                                       )


# 预处理
p = Preprocessor(train_file, seg_train_file)
p.generate_seg_file()
p.generate_catagory_dict(category_file)
p.generate_vocab_file(vocab_file)

tf.logging.set_verbosity(tf.logging.INFO)

hps = get_default_params()

vocab = Vocab(vocab_file, hps.num_word_threshold)
category_vocab = CategoryDict(category_file)

vocab_size = vocab.size()
num_classes = category_vocab.size()

train_dataset = TextDataSet(seg_train_file, vocab, category_vocab, hps.num_timesteps, False)
text_model = TextModel(hps, vocab_size, num_classes)

placeholders, metrics, others = text_model.create_model()
inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others

init_op = tf.global_variables_initializer()

train_keep_prob_value = 0.8

num_train_steps = 10000
saver = tf.train.Saver()
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    time_start = time.time()
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs, batch_labels = train_dataset.next_batch(hps.batch_size, True)
        outputs_val = sess.run([loss, accuracy, train_op, global_step],
                               feed_dict={inputs: batch_inputs, outputs: batch_labels,
                                          keep_prob: train_keep_prob_value})
        loss_val, accuracy_val, _, global_step_val = outputs_val
        if global_step_val % 10 == 0:
            time_end = time.time()
            tf.logging.info("Step: %5d, loss: %3.3f, accuracy: %3.3f, time cost: %.3f" % (
            global_step_val, loss_val, accuracy_val, time_end - time_start))
            saver.save(sess, "./model/text_classify.ckpt")
