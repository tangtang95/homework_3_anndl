import tensorflow as tf
import os
from datetime import datetime

from src.data.CustomDataGenerator import get_number_of_labels


class ConvRecurrentNetwork(object):
    EMBEDDING_SIZE = 50

    def get_image_model(self, img_h, img_w):
        model = tf.keras.models.Sequential()
        depth_max_pool = 4
        total_depth = 8

        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                                         input_shape=(img_h, img_w, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        for i in range(1, total_depth - 1):
            model.add(tf.keras.layers.Conv2D(filters=32 * i, kernel_size=(3, 3), padding='same', activation='relu'))
            if i <= depth_max_pool:
                model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.GlobalAveragePooling2D())
        return model

    def get_question_model(self, question_len, wtoi):
        question_input = tf.keras.layers.Input(shape=question_len)
        lstm_model = tf.keras.layers.Embedding(len(wtoi) + 1, self.EMBEDDING_SIZE,
                                               input_length=question_len)(question_input)
        lstm_model = tf.keras.layers.LSTM(128, return_state=False)(lstm_model)
        lstm_model = tf.keras.Model(inputs=question_input, outputs=lstm_model)
        return lstm_model

    def get_model(self, question_len, wtoi, img_h, img_w, seed):
        cnn_model = self.get_image_model(img_h, img_w)
        lstm_model = self.get_question_model(question_len, wtoi)

        model = tf.keras.layers.concatenate([cnn_model.output, lstm_model.output])
        model = tf.keras.layers.Dense(units=128)(model)
        model = tf.keras.layers.Dropout(0.2, seed=seed)(model)
        model = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(model)

        model = tf.keras.Model(inputs=[lstm_model.input, cnn_model.input], outputs=model)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


def get_callbacks(root_path, model_name, save_checkpoint=True, save_logs=True, early_stop=False):
    exps_dir = os.path.join(root_path, 'segmentation_experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks = []

    # Model checkpoint
    # ----------------
    if save_checkpoint:
        ckpt_dir = os.path.join(exp_dir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                           save_weights_only=True)  # False to save the model directly
        callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    # ---------------------------------
    if save_logs:
        tb_dir = os.path.join(exp_dir, 'tb_logs')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)

        # By default shows losses and metrics for both training and validation
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                     profile_batch=0,
                                                     histogram_freq=0)  # if 1 shows weights histograms
        callbacks.append(tb_callback)

    # Early Stopping
    # --------------
    if early_stop:
        es_callback = tf.keras.callback.EarlyStopping(monitor='val_loss', patience=10)
        callbacks.append(es_callback)
