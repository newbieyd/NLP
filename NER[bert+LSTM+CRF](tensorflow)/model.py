import tensorflow as tf
from config import FLAGS
import time
from other import time_convert, save_to_file
from bert import modeling, tokenization

def make_dict(batch_data, max_len, token):
    word_ids_list = list()
    word_mask_list = list()
    word_segment_ids_list = list()
    i = 0
    for sentence in batch_data:
        split_tokens = token.tokenize(sentence)
        split_tokens.insert(0, '[CLS]')
        split_tokens.append('[SEP]')
        while len(split_tokens) < max_len:
            split_tokens.append('[PAD]')
        word_ids = token.convert_tokens_to_ids(split_tokens)
        word_mask = [1] * len(word_ids)
        word_segment_ids = [0] * len(word_ids)
        word_ids_list.append(word_ids)
        word_mask_list.append(word_mask)
        word_segment_ids_list.append(word_segment_ids)
        i += 1
    return word_ids_list, word_mask_list, word_segment_ids_list


def bert():
    bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")# 配置文件地址。
    input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
    segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")
    model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
    saver = tf.train.Saver()
    output = model.get_sequence_output()
    return  input_ids, input_mask, segment_ids, saver, output

#LSTM定义
def LSTM(tag_num):
    #定义占位符
    x = tf.placeholder("float", [None, None, 768])
    y = tf.placeholder("float", [None, None, tag_num])
    length = tf.placeholder("int32", [None])
    current_epoch = tf.placeholder("int32")
    #构建单层LSTM
    fw_cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden)
    bw_cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length ,dtype=tf.float32)

    y_predict = tf.concat(outputs, 2)
    y_predict = tf.contrib.layers.fully_connected(y_predict, tag_num)

    y_label = tf.cast(tf.argmax(y, axis=2), tf.int32)
    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(y_predict, y_label, length)
    loss = tf.reduce_mean(-log_likelihood)
    learning_rate = FLAGS.learning_rate
    # learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate, global_step=current_epoch,
    #                                            decay_steps=FLAGS.decay_epoch, decay_rate=FLAGS.decay_learning_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    y_label_pred, _ = tf.contrib.crf.crf_decode(y_predict, transition, length)

    return loss, optimizer, x, y, length, y_label_pred, y_label, learning_rate, current_epoch

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



#训练过程
def training(train_data, train_label_list, train_length_list, dev_data, dev_label_list, dev_length_list, tag_list, max_len):
    pathname = "chinese_L-12_H-768_A-12/bert_model.ckpt"  # 模型地址
    bert_vocab_file = "./chinese_L-12_H-768_A-12/vocab.txt"
    input_ids, input_mask, segment_ids, saver_bert, output = bert()
    step = len(train_data)
    show = (step // FLAGS.show_step) + 1
    cost, optimizer, x, y, length, y_label_pred, y_label, learning_rate, current_epoch = LSTM(len(tag_list))
    saver_ner = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    # 启动session
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())  # 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。
        saver_bert.restore(sess, pathname)
        token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
        best_f1 = -1
        loop_flag = False
        restore_flag = False
        epoch = 0
        max_epoch = FLAGS.max_epoch
        while epoch < max_epoch:
            epoch += 1
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
            for i in range(len(train_data)):
                batch_sentences = train_data[i]
                batch_y = train_label_list[i]
                batch_length = train_length_list[i]
                word_ids_list, word_mask_list, word_segment_ids_list = make_dict(batch_sentences, max_len, token)
                fd = {input_ids: word_ids_list, input_mask: word_mask_list, segment_ids: word_segment_ids_list}
                batch_x = sess.run(output, feed_dict=fd)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, length: batch_length, current_epoch:epoch})
                loss, pred, label = sess.run([cost, y_label_pred, y_label], feed_dict={x: batch_x, y: batch_y, length: batch_length, current_epoch:epoch})
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
                    print("Epoch:", epoch, "Iter:", i*FLAGS.batch+len(batch_x), " Loss = " + "{:.6f}".format(loss),
                          " Accuracy = " + "{:.4f}".format(acc), " Precision = " + "{:.4f}".format(precision),
                          " Recall= " + "{:.4f}".format(recall), " F1 = " + "{:.4f}".format(f1), "time:", time_convert(t_end-t_start))
                    t_start = time.time()
                    pred_list = list()
                    label_list = list()
                    length_list = list()
            t_end = time.time()
            acc, pre_num, gold, right, _ = statistics(all_pred_list, all_label_list, all_length_list, tag_list, False)
            precision, recall, f1 = calculate(pre_num, gold, right)
            print("Epoch:", epoch, "train Accuracy = " + "{:.4f}".format(acc), " Precision = " + "{:.4f}".format(precision),
                  " Recall = " + "{:.4f}".format(recall), " F1 = " + "{:.4f}".format(f1),
                  "time:", time_convert(t_end-t_start_epoch))

            #dev
            all_pred_list = list()
            all_label_list = list()
            all_length_list = list()
            t_start = time.time()
            for i in range(len(dev_data)):
                batch_sentences = dev_data[i]
                batch_y = dev_label_list[i]
                batch_length = dev_length_list[i]
                word_ids_list, word_mask_list, word_segment_ids_list = make_dict(batch_sentences, max_len, token)
                fd = {input_ids: word_ids_list, input_mask: word_mask_list, segment_ids: word_segment_ids_list}
                batch_x = sess.run(output, feed_dict=fd)
                loss, pred, label = sess.run([cost, y_label_pred, y_label], feed_dict={x: batch_x, y: batch_y, length: batch_length, current_epoch: epoch})
                all_pred_list.append(pred)
                all_label_list.append(label)
                all_length_list.append(batch_length)
            acc, precision, gold, right, entity_dict = statistics(all_pred_list, all_label_list, all_length_list, tag_list, True)
            precision, recall, dev_f1 = calculate(precision, gold, right)
            t_end = time.time()
            print("Epoch:", epoch, "dev Accuracy = " + "{:.4f}".format(acc), " Precision = " + "{:.4f}".format(precision),
                  " Recall = " + "{:.4f}".format(recall), " F1 = " + "{:.4f}".format(dev_f1), "time:", time_convert(t_end - t_start))
            for entity in list(entity_dict):
                pre_num, gold, right = entity_dict[entity]
                precision, recall, f1 = calculate(pre_num, gold, right)
                print(entity, " Precision = " + "{:.4f}".format(precision), " Recall = " + "{:.4f}".format(recall),
                      " F1 = " + "{:.4f}".format(f1), "precision:" + str(pre_num), "gold:" + str(gold), "right:" + str(right))
            # dev集上获得更高的F值则保存模型， 并再后面存储test集的结果
            if dev_f1 > best_f1:
                loop_flag
                best_f1 = dev_f1
                saver_ner.save(sess, FLAGS.model)
                print("better result!\nsave model!")
            # if epoch % FLAGS.decay_epoch == 0:= True
            #                 restore_flag = True
            #     if restore_flag:
            #         print("reload the best model!")
            #         saver_ner.restore(sess, FLAGS.model)
            #     if loop_flag and epoch == max_epoch:
            #         max_epoch += FLAGS.decay_epoch
            #     loop_flag = False
    print("Training finish!")
