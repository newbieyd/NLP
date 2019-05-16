from keras.models import *
from keras.layers import *
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

flag_tag_list_file = "./model/tag_list"
flag_vocabulary_file = "./model/vocabulary"
flag_model_file = "./model/model.h5"
flag_char_dim = 300
flag_filters = 100
flag_kernel_size = 3
flag_batch_size = 500
flag_max_len = 300


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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


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


def classification_test(sentences):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    tag_list = load_tag_list(flag_tag_list_file)
    print("load tag list!")
    char_dict = load_vocabulary(flag_vocabulary_file)
    print("load vocabulary!")
    token = keras.preprocessing.text.Tokenizer(filters='', char_level=True, oov_token='<UNK>')
    token.word_index = char_dict
    x_sequences = token.texts_to_sequences(sentences)
    x = keras.preprocessing.sequence.pad_sequences(sequences=x_sequences, maxlen=flag_max_len, padding='post',truncating='post', value=0)
    model = classfication_model(len(tag_list), flag_kernel_size, len(char_dict) + 1, flag_char_dim, flag_filters)
    model.load_weights(flag_model_file)
    preds = model.predict(x, batch_size=flag_batch_size)
    preds = np.argmax(preds, axis=1)
    pred_labels = list()
    for p in preds:
        pred_labels.append(tag_list[p])
    results = list()
    for i in range(len(sentences)):
        results.append((sentences[i], pred_labels[i]))
    return results, tag_list


def main():
    sentences = list()
    sentences.append("这是一个测试")
    results = classification_test(sentences)
    print(results)


if __name__ == "__main__":
    main()