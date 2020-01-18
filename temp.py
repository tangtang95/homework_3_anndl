from src.data.CustomDataset import read_train_valid_data, read_test_data
import os
import math
import tensorflow as tf

from src.data.CustomTokenizer import CustomTokenizer
from src.model.models import ConvRecurrentNetwork
from src.submission.submission import predict_submissions
from src.utils.utils import get_seed

if __name__ == '__main__':
    batch_size = 16
    img_h = 32
    img_w = 32

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LSTM(32))


    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()
