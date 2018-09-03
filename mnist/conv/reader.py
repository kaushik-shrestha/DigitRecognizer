import pandas as pd
import keras.utils as util
from .. import config as cfg


def read_train():
    train = pd.read_csv(cfg.root + "data/train.csv")
    y_train = util.to_categorical(train['label'])
    x_train = normalize(train.drop(labels=['label'], axis=1))
    del train
    return x_train, y_train


def read_test():
    test = pd.read_csv(cfg.root + "data/test.csv")
    return normalize(test)


def normalize(x):
    x = x.values.reshape(-1, 28, 28, 1)
    x = x / 255.0
    return x
