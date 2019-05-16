from config import FLAGS
from data_processing import load_train_file, load_dev_file, label_processing
from model import training
from other import save_to_file
import sys

def main():
    max_len = 0
    min_len = sys.maxsize
    print("loading train file...")
    train_sentence_list, train_label_list, sentence_num, max_l, min_l, words_num, tag2id = load_train_file(FLAGS.train_file)
    if max_l > max_len:
        max_len = max_l
    if min_l < min_len:
        min_len = min_l
    print("train data sentences is", sentence_num, "and the number of all words is", words_num)
    print("loading dev file...")
    dev_sentence_list, dev_label_list, sentence_num, max_l, min_l, words_num= load_dev_file(FLAGS.dev_file, tag2id)
    if max_l > max_len:
        max_len = max_l
    if min_l < min_len:
        min_len = min_l
    print("dev data sentences is", sentence_num, "and the number of all words is", words_num)
    print("The maximum sentence length is", max_len)
    print("The minimum sentence length is", min_len)
    max_len += 2
    train_label_list, train_length_list = label_processing(train_label_list, max_len, tag2id)
    dev_label_list, dev_length_list = label_processing(dev_label_list, max_len, tag2id)
    id2tag = dict()
    for tag in list(tag2id):
        id2tag[tag2id[tag]] = tag
    tag_list = list()
    for i in range(len(list(id2tag))):
        tag_list.append(id2tag[i])
    data = list()
    for t in tag_list:
        data.append(t + '\n')
    save_to_file(FLAGS.tag_list, data)
    print("save tag_list!")
    training(train_sentence_list, train_label_list, train_length_list, dev_sentence_list, dev_label_list, dev_length_list, tag_list, max_len)


if __name__ == "__main__":
    main()