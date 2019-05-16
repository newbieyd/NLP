from keras.models import *
from keras.layers import *
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

flag_train_file = "./data/loc_class.train"
flag_dev_file = "./data/loc_class.dev"
flag_tag_list_file = "./model/tag_list"
flag_vocabulary_file = "./model/vocabulary"
flag_model_file = "./model/ner.h5"
flag_char_dim = 300
flag_filters = 100
flag_kernel_size = 3
flag_batch_size = 5
flag_max_epoch = 100


def load_file(path):
    with open(path, "r", encoding="utf8") as fin:
        sentences = list()
        labels = list()
        tag_list = list()
        max_len = 0
        for line in fin.readlines():
            temp = line[:-1].split('\t')
            sentences.append(temp[1])
            labels.append(temp[0])
            if temp[0] not in tag_list:
                tag_list.append(temp[0])
            if len(temp[1]) > max_len:
                max_len = len(temp[1])
    return sentences, labels, max_len, tag_list


def label_to_id(labels, tag_list):
    tag_dict = dict()
    i = 0
    for t in tag_list:
        tag_dict[t] = i
        i += 1
    result = list()
    for lable in labels:
        result.append(tag_dict[lable])
    return result



def save_to_file(path, data):
    with open(path, "w", encoding="utf8") as fout:
        for d in data:
            fout.write(d)


def classfication_model(class_num,kernel_size,max_features,embedding_dims,filters = 250):
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims))  # 使用Embedding层将每个词编码转换为词向量
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    # 池化
    model.add(GlobalMaxPooling1D())
    model.add(Dense(class_num, activation='softmax'))  # 第一个参数units: 全连接层输出的维度，即下一层神经元的个数。
    model.add(Dropout(0.2))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

best_acc = 0
class Evaluate(keras.callbacks.Callback):
    def __init__(self, sequences, labels, tag_list, model):
        self.x = sequences
        self.labels = labels
        self.tag_list = tag_list
        self.model = model

    def on_epoch_end(self, batch, logs={}):
        preds = self.model.predict(self.x, batch_size=flag_batch_size)
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(self.labels, axis=1)
        self.evaluate(preds, labels)

    def evaluate(self, preds, labels):
        right_num, class_dict = self.statistics(preds, labels)
        # print(preds, labels)
        # print(right_num, class_dict)
        all_num = labels.shape[0]
        acc = right_num/all_num
        print("dev:", "acc = " + "{:.4f}".format(acc), " all:", all_num, "right:", right_num)
        for c in list(class_dict):
            pre_num, gold, right = class_dict[c]
            precision, recall, f1 = self.calculate(pre_num, gold, right)
            print(self.tag_list[c], " Precision = " + "{:.4f}".format(precision), " Recall = " + "{:.4f}".format(recall),
                  " F1 = " + "{:.4f}".format(f1), "precision:" + str(pre_num), "gold:" + str(gold), "right:" + str(right))
        global best_acc
        if best_acc < acc or best_acc == 0:
            best_acc = acc
            self.model.save(flag_model_file)
            print("better result! save model!")


    def statistics(self, preds, labels):
        right = 0
        class_dict = dict()
        for i in range(len(labels)):
            pred = preds[i]
            label = labels[i]
            if pred in class_dict:
                class_dict[pred][0] += 1
            else:
                class_dict[pred] = [1, 0, 0]
            if label in class_dict:
                class_dict[label][1] += 1
            else:
                class_dict[label] = [0, 1, 0]
            if pred == label:
                right += 1
                class_dict[label][2] += 1
        return right, class_dict

    def calculate(self, predict, gold, correct):
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
        return precision, recall, f1

    def on_train_end(self, logs={}):
        self.model.load_weights(flag_model_file)
        print("reload best model!")
        print("best result:")
        preds = self.model.predict(self.x, batch_size=flag_batch_size)
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(self.labels, axis=1)
        self.evaluate(preds, labels)


def main():
    train_sentences, train_labels, max_len, tag_list = load_file(flag_train_file)
    print("load trian data, sentences:", len(train_sentences), "maximum length:", max_len)
    dev_sentences, dev_labels, _, _ = load_file(flag_dev_file)
    print("load dev data, sentences:", len(dev_sentences))
    data = list()
    for t in tag_list:
        data.append(t + '\n')
    save_to_file(flag_tag_list_file, data)
    print("save tag_list!")
    token = keras.preprocessing.text.Tokenizer(filters='', char_level=True, oov_token='<UNK>')
    token.fit_on_texts(train_sentences)
    char_dict = token.word_index
    data = list()
    for c in list(char_dict):
        data.append(c + '\t' + str(char_dict[c]) + '\n')
    save_to_file(flag_vocabulary_file, data)
    print("save vocabulary!")
    train_x_sequences = token.texts_to_sequences(train_sentences)
    train_x = keras.preprocessing.sequence.pad_sequences(sequences=train_x_sequences, maxlen=max_len, padding='post',truncating='post', value=0)
    dev_x_sequences = token.texts_to_sequences(dev_sentences)
    dev_x = keras.preprocessing.sequence.pad_sequences(sequences=dev_x_sequences, maxlen=max_len, padding='post', truncating='post', value=0)
    train_labels = label_to_id(train_labels, tag_list)
    train_labels = keras.utils.to_categorical(train_labels, num_classes=len(tag_list))
    dev_labels = label_to_id(dev_labels, tag_list)
    dev_labels = keras.utils.to_categorical(dev_labels, num_classes=len(tag_list))
    model = classfication_model(len(tag_list), flag_kernel_size, len(char_dict)+1, flag_char_dim, flag_filters)
    evaluate = Evaluate(dev_x, dev_labels, tag_list, model)
    model.fit(train_x, train_labels, verbose=2, batch_size=flag_batch_size, epochs=flag_max_epoch, callbacks=[evaluate])

if __name__ == "__main__":
    main()
