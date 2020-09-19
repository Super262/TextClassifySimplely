import tensorflow as tf
import time
from Essentials import Vocab, CategoryDict, TextDataSet, TextModel, Preprocessor

test_file = './data/test.txt'
seg_test_file = './data/test.seg.txt'
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

# 预处理
p = Preprocessor(test_file, seg_test_file)
p.generate_seg_file()

tf.logging.set_verbosity(tf.logging.INFO)

hps = get_default_params()

vocab = Vocab(vocab_file, hps.num_word_threshold)
category_vocab = CategoryDict(category_file)

vocab_size = vocab.size()
num_classes = category_vocab.size()

test_dataset = TextDataSet(seg_test_file, vocab, category_vocab, hps.num_timesteps, False)
text_model = TextModel(hps, vocab_size, num_classes)

placeholders, metrics, others = text_model.create_model()
inputs, outputs, keep_prob = placeholders
accuracy = metrics[1]
data_size = test_dataset.size()

init_op = tf.global_variables_initializer()

test_keep_prob_value = 1.0

saver = tf.train.Saver()
with tf.Session() as sess:
    time_start = time.time()
    saver.restore(sess, "./model/text_classify.ckpt")
    outputs_val = 0.0
    for i in range(data_size):
        batch_inputs, batch_labels = test_dataset.next_batch(hps.batch_size, False)
        outputs_val += float(sess.run(accuracy,
                                      feed_dict={inputs: batch_inputs, outputs: batch_labels,
                                                 keep_prob: test_keep_prob_value}))
        if (i + 1) % 10 == 0:
            time_end = time.time()
            print("The accuracy now is %.3f, time cost: %.3f" % (outputs_val / (i + 1), time_end - time_start))

    print("The accuracy on test set is %.3f" % (outputs_val / data_size))

    