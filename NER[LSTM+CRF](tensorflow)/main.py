import tensorflow as tf
import numpy as np
import os
import time
import sys

flags = tf.app.flags
flags.DEFINE_string("train_file",       os.path.join("data", "PER.NER.train"),     "Path for train data")
flags.DEFINE_string("dev_file",         os.path.join("data", "PER.NER.dev"),       "Path for dev data")
flags.DEFINE_string("model",             os.path.join("model", "model"),          "Path for model")
flags.DEFINE_string("tag_list_file",    os.path.join("model", "tag_list"),  "Path for result of test")
flags.DEFINE_string("char2id_file",        os.path.join("model", "id"),           "Path for char2id")

flags.DEFINE_integer("max_epoch",       100,             "maximum training epochs")
flags.DEFINE_integer("char_dim",        300,            "Embedding size for characters")
flags.DEFINE_integer("batch",           300,            "the size of batch")
flags.DEFINE_integer("show_step",       10,             "the number time to show every epoch")
flags.DEFINE_integer("hidden",          1024,            "the number of cells every LSTM")
flags.DEFINE_float("learning_rate",     0.001,          "Initial learning rate")
flags.DEFINE_integer("decay_epoch",          10 ,            "Degraded learning rate per decay_epoch number")
flags.DEFINE_float("decay_learning_rate",     0.9,          "Degraded learning rate")
flags.DEFINE_integer("tag_column",         2,            "The number is the column of tag, which start at 0")
FLAGS = tf.app.flags.FLAGS


#时间转换
def time_convert(t):
    sencod = int(t) % 60
    result = str(sencod) + 's'
    minute = int(t) // 60
    if minute > 0:
        hour = minute // 60
        minute = minute % 60
        result = str(minute) + 'min ' + result
        if hour > 0:
            result = str(hour) + 'h ' + result
    else:
        result = "{:.2f}".format(t) + 's'
    return result

def save_to_file(path, data):
    with open(path, "w", encoding="utf8") as fout:
        for d in data:
            fout.write(d)

def load_file(path):
    result = list()
    max_len = 0
    min_len = sys.maxsize
    words_num = 0
    tag_list = list()
    tag_list.append('O')
    with open(path, "r", encoding="utf8") as fin:
        sentence = list()
        for line in fin.readlines():
            if line != '\n':
                temp = line[:-1].split('\t')
                words_num += 1
                if temp[FLAGS.tag_column] not in tag_list:
                    tag_list.append(temp[FLAGS.tag_column])
                sentence.append(temp)
            else:
                if len(sentence) > 0:
                    result.append(sentence)
                    if len(sentence) > max_len:
                        max_len = len(sentence)
                    if len(sentence) < min_len:
                        min_len = len(sentence)
                    sentence = list()
    if len(sentence) > 0:
        result.append(sentence)
        if len(sentence) > max_len:
            max_len = len(sentence)
        if len(sentence) < min_len:
            min_len = len(sentence)
    return result, max_len, min_len, words_num, tag_list


def make_char2id(data):
    char_dict = dict()
    for sentence in data:
        for char in sentence:
            if char[0] in char_dict:
                char_dict[char[0]] += 1
            else:
                char_dict[char[0]] = 1
    all_char_num = len(char_dict)
    char_list = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
    char2id_dict = dict()
    char2id_dict["<NULL>"] = 0
    char2id_dict["<s>"] = 1
    char2id_dict["<e>"] = 2
    i = 3
    j = 0
    while j < len(char_list):
        char2id_dict[char_list[j][0]] = i
        i += 1
        j += 1
    return char2id_dict, all_char_num


def make_dataset_char2id(data, id, max_len, tag_list):
    tag_embedding = np.eye(len(tag_list))
    result = list()
    x = np.zeros(shape=(FLAGS.batch ,max_len), dtype=np.int)
    y = np.zeros(shape=(FLAGS.batch, max_len, len(tag_list)), dtype=np.float)
    length = list()
    i = 0
    for d in data:
        length.append(len(d) + 2)
        s_x = np.zeros(shape=(max_len), dtype=np.int)
        s_y = np.zeros(shape=(max_len, len(tag_list)), dtype=np.float)
        s_x[0] = id["<s>"]
        s_y[0] = tag_embedding[tag_list.index('O')]
        j = 0
        while j < len(d):
            ch = d[j][0]
            tag = d[j][FLAGS.tag_column]
            if ch in id:
                s_x[j+1] = id[ch]
                s_y[j+1] = tag_embedding[tag_list.index(tag)]
            else:
                s_x[j+1] = id["<NULL>"]
                s_y[j+1] = tag_embedding[tag_list.index(tag)]
            j += 1
        s_x[j+1] = id["<e>"]
        s_y[j+1] = tag_embedding[tag_list.index('O')]
        x[i] = s_x
        y[i] = s_y
        i += 1
        if i >= FLAGS.batch:
            result.append((x, y, np.array(length)))
            x = np.zeros(shape=(FLAGS.batch, max_len), dtype=np.int)
            y = np.zeros(shape=(FLAGS.batch, max_len, len(tag_list)), dtype=np.float)
            length = list()
            i = 0
    if i > 0:
        rest = len(data) % FLAGS.batch
        rest_x = np.zeros(shape=(rest, max_len), dtype=np.int)
        rest_y = np.zeros(shape=(rest, max_len, len(tag_list)), dtype=np.float)
        for j in range(rest):
            rest_x[j] = x[j]
            rest_y[j] = y[j]
        result.append((rest_x, rest_y, np.array(length)))
    return result


def change_to_site(sentence, length, tag_list):
    result = list()
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
                result.append((start, end, tag_type))
        j += 1
    return result


def statistics(pred_list, label_list, length_list, tag_list, entity_flag):
    acc_num = 0
    acc_right = 0
    label_num = 0
    precision = 0
    right = 0
    entity_dict = dict()
    for i in range(len(length_list)):
        for j in range(len(length_list[i])):
            pred = pred_list[i][j]
            label = label_list[i][j]
            length = length_list[i][j]
            for k in range(length):
                acc_num += 1
                if pred[k] == label[k]:
                    acc_right += 1
            pred = change_to_site(pred, length, tag_list)
            label = change_to_site(label, length, tag_list)
            precision += len(pred)
            label_num += len(label)
            if entity_flag:
                for p in pred:
                    if p[2] not in entity_dict:
                        entity_dict[p[2]] = [1, 0, 0]
                    else:
                        entity_dict[p[2]][0] += 1
                for l in label:
                    if l[2] not in entity_dict:
                        entity_dict[l[2]] = [0, 1, 0]
                    else:
                        entity_dict[l[2]][1] += 1
            for p in pred:
                if p in label:
                    right += 1
                    if entity_flag:
                        entity_dict[p[2]][2] += 1
    acc = acc_right/acc_num
    return  acc, precision, label_num, right, entity_dict


#计算准确率， 召回率， F值
def calculate(predict, gold, correct):
    if predict == 0:
        precision = -1
    else:
        precision = correct / predict
    if gold == 0:
        recall = -1
    else:
        recall = correct / gold
    if precision == -1 or recall == -1 or precision + recall == 0:
        f1 = -1
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return  precision, recall, f1


def main():
    t_start = time.time()
    max_len = 0
    min_len = sys.maxsize
    print("loading train file...")
    train_data, max_l, min_l, words_num, tag_list = load_file(FLAGS.train_file)
    if max_l > max_len:
        max_len = max_l
    if min_l < min_len:
        min_len = min_l
    t_end = time.time()
    print("train data sentences is", len(train_data), "and the number of all words is", words_num, "time:", time_convert(t_end - t_start))
    print("loading validation file...")
    t_start = time.time()
    dev_data, max_l, min_l, words_num, _ = load_file(FLAGS.dev_file)
    if max_l > max_len:
        max_len = max_l
    if min_l < min_len:
        min_len = min_l
    t_end = time.time()
    print("validation data sentences is", len(dev_data), "and the number of all words is", words_num, "time:", time_convert(t_end - t_start))
    print("The maximum sentence length is", max_len)
    print("The minimum sentence length is", min_len)
    print("The number of tag is", len(tag_list))
    max_len += 2
    save_data = list()
    for t in tag_list:
        data = t + '\n'
        save_data.append(data)
    save_to_file(FLAGS.tag_list_file, save_data)
    print("Save tag!")
    print("making char2id ...")
    char2id, all_char_num = make_char2id(train_data)
    print("The char number is", all_char_num)
    print("The char2id number is", len(char2id))
    save_data = list()
    for char in list(char2id):
        data = char + '\t' + str(char2id[char]) + '\n'
        save_data.append(data)
    save_to_file(FLAGS.char2id_file, save_data)
    print("Save char2id!")
    print("making dataset...")
    time.time()
    train_dataset = make_dataset_char2id(train_data, char2id, max_len, tag_list)
    dev_dataset = make_dataset_char2id(dev_data, char2id, max_len, tag_list)
    t_end = time.time()
    print("sucess making dataset! time:", time_convert(t_end - t_start))
    char_embedding = tf.get_variable(name="char_embedding", dtype=tf.float32, shape=[len(char2id), FLAGS.char_dim])
    x = tf.placeholder("int32", [None, None])
    inputs = tf.nn.embedding_lookup(char_embedding, x)
    y = tf.placeholder("float", [None, None, len(tag_list)])
    length = tf.placeholder("int32", [None])
    current_epoch = tf.placeholder("int32")
    fw_cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden)
    bw_cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, sequence_length=length, dtype=tf.float32)
    y_predict = tf.concat(outputs, 2)
    y_predict = tf.contrib.layers.fully_connected(y_predict, len(tag_list))
    y_label = tf.cast(tf.argmax(y, axis=2), tf.int32)
    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(y_predict, y_label, length)
    loss = tf.reduce_mean(-log_likelihood)
    learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate, global_step=current_epoch,
                                               decay_steps=FLAGS.decay_epoch, decay_rate=FLAGS.decay_learning_rate,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    y_label_pred, _ = tf.contrib.crf.crf_decode(y_predict, transition, length)
    best_f1 = 0
    step = len(train_dataset)
    show = (step // FLAGS.show_step) + 1
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    # 启动session
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        reload_flag = False
        lr_epoch = FLAGS.decay_epoch - 1
        for epoch in range(FLAGS.max_epoch):
            lr_epoch += 1
            #train
            t_start_epoch = time.time()
            t_start = time.time()
            #初始化统计数据
            all_pred_list = list()
            all_label_list = list()
            all_length_list = list()
            pred_list = list()
            label_list = list()
            length_list = list()
            if epoch % FLAGS.decay_epoch == 0: 
                if reload_flag:
                    print("reload the best model!")
                    saver.restore(sess, FLAGS.model)
                else:
                    lr_epoch -= 10
            for i in range(step):
                batch_x, batch_y, batch_length = train_dataset[i] #获得batch数据
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, length: batch_length, current_epoch:epoch})
                cost, pred, label, lr = sess.run([loss, y_label_pred, y_label, learning_rate], feed_dict={x: batch_x, y: batch_y, length: batch_length, current_epoch:lr_epoch})
                pred_list.append(pred)
                label_list.append(label)
                length_list.append(batch_length)
                all_pred_list.append(pred)
                all_label_list.append(label)
                all_length_list.append(batch_length)
                if i % show == 0:
                    acc, precision, gold, right,_ = statistics(pred_list, label_list, length_list, tag_list, False)
                    precision, recall, f1 = calculate(precision, gold, right)
                    t_end = time.time()
                    print("Epoch:", epoch+1, "Iter:", i*FLAGS.batch+len(batch_x), " Loss = " + "{:.6f}".format(cost),
                          " Accuracy = " + "{:.4f}".format(acc), " Precision = " + "{:.4f}".format(precision),
                          " Recall= " + "{:.4f}".format(recall), " F1 = " + "{:.4f}".format(f1), "time:", time_convert(t_end-t_start))
                    pred_list = list()
                    label_list = list()
                    length_list = list()
                    t_start = time.time()
            t_end = time.time()
            acc, pre_num, gold, right, _ = statistics(all_pred_list, all_label_list, all_length_list, tag_list, False)
            precision, recall, f1 = calculate(pre_num, gold, right)
            print("Epoch:", epoch + 1, "train Accuracy = " + "{:.4f}".format(acc), " Precision = " + "{:.4f}".format(precision),
                  " Recall = " + "{:.4f}".format(recall), " F1 = " + "{:.4f}".format(f1), "learning_rate=" + "{:.6f}".format(lr),
                  "time:", time_convert(t_end-t_start_epoch))

            #dev
            all_pred_list = list()
            all_label_list = list()
            all_length_list = list()
            t_start = time.time()
            for i in range(len(dev_dataset)):
                batch_x, batch_y, batch_length = dev_dataset[i]  # 获得batch数据
                pred, label, _ = sess.run([y_label_pred, y_label, learning_rate], feed_dict={x: batch_x, y: batch_y, length: batch_length, current_epoch: epoch})
                all_pred_list.append(pred)
                all_label_list.append(label)
                all_length_list.append(batch_length)
            acc, precision, gold, right, entity_dict = statistics(all_pred_list, all_label_list, all_length_list, tag_list, True)
            precision, recall, dev_f1 = calculate(precision, gold, right)
            t_end = time.time()
            print("Epoch:", epoch + 1, "dev Accuracy = " + "{:.4f}".format(acc), " Precision = " + "{:.4f}".format(precision),
                  " Recall = " + "{:.4f}".format(recall), " F1 = " + "{:.4f}".format(dev_f1), "time:", time_convert(t_end - t_start))
            for entity in list(entity_dict):
                pre_num, gold, right = entity_dict[entity]
                precision, recall, f1 = calculate(pre_num, gold, right)
                print(entity, " Precision = " + "{:.4f}".format(precision), " Recall = " + "{:.4f}".format(recall),
                      " F1 = " + "{:.4f}".format(f1), "precision:" + str(pre_num), "gold:" + str(gold), "right:" + str(right))
            # dev集上获得更高的F值则保存模型
            if dev_f1 > best_f1 or best_f1 == 0:
                best_f1 = dev_f1
                saver.save(sess, FLAGS.model)
                print("better result, save model!")
                reload_flag = True
        print("Training finish!")
        print("reload the best model!")
        saver.restore(sess, FLAGS.model)
        all_pred_list = list()
        all_label_list = list()
        all_length_list = list()
        for i in range(len(dev_dataset)):
            batch_x, batch_y, batch_length = dev_dataset[i]  # 获得batch数据
            pred, label, _ = sess.run([y_label_pred, y_label, learning_rate], feed_dict={x: batch_x, y: batch_y, length: batch_length, current_epoch: 0})
            all_pred_list.append(pred)
            all_label_list.append(label)
            all_length_list.append(batch_length)
        acc, precision, gold, right, entity_dict = statistics(all_pred_list, all_label_list, all_length_list, tag_list, True)
        precision, recall, dev_f1 = calculate(precision, gold, right)
        print("Best dev Accuracy = " + "{:.4f}".format(acc), " Precision = " + "{:.4f}".format(precision),
              " Recall = " + "{:.4f}".format(recall), " F1 = " + "{:.4f}".format(dev_f1))
        for entity in list(entity_dict):
            pre_num, gold, right = entity_dict[entity]
            precision, recall, f1 = calculate(pre_num, gold, right)
            print(entity, " Precision = " + "{:.4f}".format(precision), " Recall = " + "{:.4f}".format(recall),
                  " F1 = " + "{:.4f}".format(f1), "precision:" + str(pre_num), "gold:" + str(gold), "right:" + str(right))


if __name__ == "__main__":
    main()
