import tensorflow as tf
import os

flags = tf.app.flags
flags.DEFINE_string("train_file",       os.path.join("data", "per.NER.train"),     "Path for train data")
flags.DEFINE_string("dev_file",         os.path.join("data", "per.NER.dev"),       "Path for dev data")
flags.DEFINE_string("tag_list_file",    os.path.join("data", "tag_list"),       "Path for tag_list")
flags.DEFINE_string("model",             os.path.join("model", "model"),          "Path for model")
flags.DEFINE_string("tag_list",         os.path.join("model", "tag_list"),          "Path for model")
flags.DEFINE_integer("max_epoch",       100,             "maximum training epochs")
flags.DEFINE_integer("batch",            200,            "the size of batch")
flags.DEFINE_integer("show_step",       10,             "the number time to show every epoch")
flags.DEFINE_integer("hidden",           1024,            "the number of cells every LSTM")
flags.DEFINE_float("learning_rate",     0.001,          "Initial learning rate")
# flags.DEFINE_integer("decay_epoch",           10,            "Degraded learning rate per decay_epoch number")
# flags.DEFINE_float("decay_learning_rate",     0.9,          "Degraded learning rate")

flags.DEFINE_integer("tag_column",     2,               "the column for tag, start from 0")

FLAGS = tf.app.flags.FLAGS
