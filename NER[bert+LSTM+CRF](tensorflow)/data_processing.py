from config import FLAGS
import numpy as np
import sys


def load_train_file(path):
    max_len = 0
    min_len = sys.maxsize
    words_num = 0
    sentence_num = 0
    sentence_list = list()
    label_list = list()
    batch_sentence_list = list()
    batch_label_list = list()
    tag2id = dict()
    tag2id['O'] = 0
    with open(path, "r", encoding="utf8") as fin:
        sentence = ""
        sentence_label_list = list()
        line = fin.readline()
        while len(line) > 0:
            if line != '\n':
                temp = line[:-1].split('\t')
                words_num += 1
                if temp[FLAGS.tag_column] not in tag2id:
                    tag2id[temp[FLAGS.tag_column]] = len(tag2id)
                sentence_label_list.append(tag2id[temp[FLAGS.tag_column]])
                sentence += temp[0]
            else:
                if len(sentence) > 0:
                    sentence_num += 1
                    batch_sentence_list.append(sentence)
                    batch_label_list.append(sentence_label_list)
                    if len(sentence) > max_len:
                        max_len = len(sentence)
                    if len(sentence) < min_len:
                        min_len = len(sentence)
                    sentence = ""
                    sentence_label_list = list()
                    if len(batch_sentence_list) == FLAGS.batch:
                        sentence_list.append(batch_sentence_list)
                        label_list.append(batch_label_list)
                        batch_sentence_list = list()
                        batch_label_list = list()
            line = fin.readline()
    if len(batch_sentence_list) > 0:
        sentence_list.append(batch_sentence_list)
        label_list.append(batch_label_list)
    return sentence_list, label_list, sentence_num, max_len, min_len, words_num, tag2id


def load_dev_file(path, tag2id):
    max_len = 0
    min_len = sys.maxsize
    words_num = 0
    sentence_num = 0
    sentence_list = list()
    label_list = list()
    batch_sentence_list = list()
    batch_label_list = list()
    with open(path, "r", encoding="utf8") as fin:
        sentence = ""
        sentence_label_list = list()
        line = fin.readline()
        while len(line) > 0:
            if line != '\n':
                temp = line[:-1].split('\t')
                words_num += 1
                sentence_label_list.append(tag2id[temp[FLAGS.tag_column]])
                sentence += temp[0]
            else:
                if len(sentence) > 0:
                    sentence_num += 1
                    batch_sentence_list.append(sentence)
                    batch_label_list.append(sentence_label_list)
                    if len(sentence) > max_len:
                        max_len = len(sentence)
                    if len(sentence) < min_len:
                        min_len = len(sentence)
                    sentence = ""
                    sentence_label_list = list()
                    if len(batch_sentence_list) == FLAGS.batch:
                        sentence_list.append(batch_sentence_list)
                        label_list.append(batch_label_list)
                        batch_sentence_list = list()
                        batch_label_list = list()
            line = fin.readline()
    if len(batch_sentence_list) > 0:
        sentence_list.append(batch_sentence_list)
        label_list.append(batch_label_list)
    return sentence_list, label_list, sentence_num, max_len, min_len, words_num


def label_processing(data, max_len, tag2id):
    length_list = list()
    result = list()
    for batch_data in data:
        batch_length = list()
        y = np.zeros(shape=(len(batch_data), max_len, len(tag2id)), dtype=np.float)
        for i in range(len(batch_data)):
            label_list = batch_data[i]
            batch_length.append(len(label_list) + 2)
            s_y = np.zeros(shape=(max_len, len(tag2id)), dtype=np.float)
            for j in range(len(label_list)):
                s_y[j + 1][label_list[j]] = 1
            y[i] = s_y
        result.append(y)
        length_list.append(batch_length)
    return result, length_list