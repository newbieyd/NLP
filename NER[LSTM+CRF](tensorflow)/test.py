import tensorflow as tf
import numpy as np


flag_model = "./model/model"
flag_tag_list_file = "./model/tag_list"
flag_char2id_file = "./model/id"

flag_char_dim = 300
flag_batch = 300
flag_hidden = 1024
flag_maxlen = 194


def make_dataset_char2id(data, id, max_len):
    result = list()
    x = np.zeros(shape=(flag_batch ,max_len), dtype=np.int)
    length = list()
    i = 0
    for d in data:
        length.append(len(d) + 2)
        s_x = np.zeros(shape=(max_len), dtype=np.int)
        s_x[0] = id["<s>"]
        j = 0
        while j < len(d):
            ch = d[j]
            if ch in id:
                s_x[j+1] = id[ch]
            else:
                s_x[j+1] = id["<NULL>"]
            j += 1
        s_x[j+1] = id["<e>"]
        x[i] = s_x
        i += 1
        if i >= flag_batch:
            result.append((x, np.array(length)))
            x = np.zeros(shape=(flag_batch, max_len), dtype=np.int)
            length = list()
            i = 0
    if i > 0:
        rest = len(data) % flag_batch
        rest_x = np.zeros(shape=(rest, max_len), dtype=np.int)
        for j in range(rest):
            rest_x[j] = x[j]
        result.append((rest_x, np.array(length)))
    return result


def change_to_site(sentences, length_list, tag_list):
    result = list()
    for i in range(sentences.shape[0]):
        sentence = sentences[i]
        length = length_list[i]
        entity_list = list()
        j = 1
        while j < length - 1:
            tag = tag_list[sentence[j]]
            if tag[0] == "B":
                tag_type = tag[2:]
                start = j
                j += 1
                tag = tag_list[sentence[j]]
                if tag[0] == "I":
                    right_flag = True
                else:
                    right_flag = False
                while tag[0] == "I":
                    if tag[2:] != tag_type:
                        right_flag = False
                    j += 1
                    tag = tag_list[sentence[j]]
                end = j
                j -= 1
                if right_flag:
                    entity_list.append((start, end, tag_type))
            j += 1
        result.append(entity_list)
    return result


def load_tag_list(path):
    tag_list = list()
    with open(path, "r", encoding="utf8") as fin:
        for line in fin.readlines():
            tag_list.append(line[:-1])
    return tag_list


def load_vocabulary(path):
    char_dict = dict()
    with open(path, "r", encoding="utf8") as fin:
        for line in fin.readlines():
            temp = line[:-1].split('\t')
            char_dict[temp[0]] = int(temp[1])
    return  char_dict


def test(sentences):
    tf.reset_default_graph()
    tag_list = load_tag_list(flag_tag_list_file)
    print("load tag list!")
    char2id = load_vocabulary(flag_char2id_file)
    print("load vocabulary!")
    x_dataset = make_dataset_char2id(sentences, char2id, flag_maxlen)
    char_embedding = tf.get_variable(name="char_embedding", dtype=tf.float32, shape=[len(char2id), flag_char_dim])
    x = tf.placeholder("int32", [None, None])
    y = tf.placeholder("float", [None, None, len(tag_list)])
    inputs = tf.nn.embedding_lookup(char_embedding, x)
    length = tf.placeholder("int32", [None])
    fw_cell = tf.contrib.rnn.LSTMCell(flag_hidden)
    bw_cell = tf.contrib.rnn.LSTMCell(flag_hidden)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, sequence_length=length, dtype=tf.float32)
    y_predict = tf.concat(outputs, 2)
    y_predict = tf.contrib.layers.fully_connected(y_predict, len(tag_list))
    y_label = tf.cast(tf.argmax(y, axis=2), tf.int32)
    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(y_predict, y_label, length)
    y_label_pred, _ = tf.contrib.crf.crf_decode(y_predict, transition, length)
    results = list()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    # 启动session
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, flag_model)
        sentence_site = 0
        for i in range(len(x_dataset)):
            batch_x, batch_length = x_dataset[i]  # 获得batch数据
            preds = sess.run(y_label_pred, feed_dict={x: batch_x, length: batch_length})
            entities_site = change_to_site(preds, batch_length, tag_list)
            for e in entities_site:
                sentence = sentences[sentence_site]
                sentence_site += 1
                entity_list = list()
                for entity in e:
                    data = (sentence[entity[0] - 1:entity[1] - 1], entity[2])
                    if data not in entity_list:
                        entity_list.append(data)
                results.append(entity_list)
    return results

def main():
    sentences = list()
    sentences.append("这是一个例子！")
    results = order_test(sentences)
    print(results)

if __name__ == "__main__":
    main()
