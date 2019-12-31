import tensorflow as tf
import os
from datetime import datetime

from src.data.CustomDataGenerator import get_number_of_labels


def _get_transfer_model_(img_h, img_w, application_name, fine_tuning):
    if application_name == "vgg16":
        model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                  weights='imagenet',
                                                  input_shape=(img_h, img_w, 3),
                                                  pooling="avg")
    elif application_name == "resnet50v2":
        model: tf.keras.Model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                                                           input_shape=(img_h, img_w, 3),
                                                                           pooling="avg")
    elif application_name == "inceptionresnetv2":
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                            input_shape=(img_h, img_w, 3),
                                                                            pooling="avg")
    else:
        raise NotImplemented("Transfer from {} model is not implemented.".format(application_name))

    if not fine_tuning:
        model.trainable = False

    return model


class TransferBidirectionalGRU(object):

    def __init__(self, embedding_size=25):
        self.EMBEDDING_SIZE = embedding_size

    def get_image_model(self, img_h, img_w, application_name="vgg16", fine_tuning=True):
        return _get_transfer_model_(img_w=img_w, img_h=img_h, application_name=application_name,
                                    fine_tuning=fine_tuning)

    def get_question_model(self, question_len, wtoi, n_units):
        # ATTENTION MODEL
        question_input = tf.keras.layers.Input(shape=question_len)
        question_embedding = tf.keras.layers.Embedding(len(wtoi) + 1, self.EMBEDDING_SIZE,
                                                       input_length=question_len)(question_input)
        bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=n_units,
                                                                          return_state=False,
                                                                          unroll=True,
                                                                          ))(question_embedding)
        bidirectional_model = tf.keras.Model(inputs=question_input, outputs=bidirectional)
        return bidirectional_model

    def get_model(self, question_len, wtoi, img_h, img_w, seed, n_units=512):
        cnn_model = self.get_image_model(img_h, img_w)
        question_model = self.get_question_model(question_len, wtoi, n_units=n_units)

        model = tf.keras.layers.concatenate([cnn_model.output, question_model.output])
        model = tf.keras.layers.Dense(units=128)(model)
        model = tf.keras.layers.Dropout(0.2, seed=seed)(model)
        model = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(model)

        model = tf.keras.Model(inputs=[question_model.input, cnn_model.input], outputs=model)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


class ConvTransferSelfAttention(object):
    def __init__(self, embedding_size=10):
        self.EMBEDDING_SIZE = embedding_size

    def get_image_model(self, img_h, img_w, application_name="vgg16", fine_tuning=True):
        return _get_transfer_model_(img_w=img_w, img_h=img_h, application_name=application_name,
                                    fine_tuning=fine_tuning)

    def get_question_model(self, question_len, wtoi, cnn_model: tf.keras.Model, n_filters_conv=100, attention_unit=128):
        # ATTENTION MODEL
        question_input = tf.keras.layers.Input(shape=question_len)
        question_embedding = tf.keras.layers.Embedding(len(wtoi) + 1, self.EMBEDDING_SIZE,
                                                       input_length=question_len)(question_input)
        cnn_layer_question = tf.keras.layers.Conv1D(filters=n_filters_conv,
                                                    kernel_size=4,
                                                    padding='same')
        question_embedding = cnn_layer_question(question_embedding)
        cnn_layer_image = tf.keras.layers.Conv2D(filters=n_filters_conv,
                                                 kernel_size=(1, 1),
                                                 padding="same")
        image_features = cnn_layer_image(cnn_model.layers[-2].output)
        query_value_attention = tf.keras.layers.Attention(attention_unit)([question_embedding, image_features])
        attention_model = tf.keras.Model(inputs=[question_input, cnn_model.input], outputs=query_value_attention)

        return attention_model

    def get_model(self, question_len, wtoi, img_h, img_w, seed, fine_tuning=True, application_name="vgg16",
                  n_conv_filters=1000, attention_unit=128):
        cnn_model = self.get_image_model(img_h, img_w, fine_tuning=fine_tuning, application_name=application_name)
        attention_model = self.get_question_model(question_len, wtoi, cnn_model, n_filters_conv=n_conv_filters,
                                                  attention_unit=attention_unit)

        # TOP NET
        top_net = tf.keras.layers.GlobalAveragePooling2D()(attention_model.output)
        top_net = tf.keras.layers.Dense(units=128)(top_net)
        top_net = tf.keras.layers.Dropout(0.2, seed=seed)(top_net)
        top_net = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(top_net)

        model = tf.keras.Model(inputs=attention_model.input, outputs=top_net)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


class ConvImageTransferLSTM(object):
    EMBEDDING_SIZE = 50

    def get_image_model(self, img_h, img_w, application_name="vgg16", fine_tuning=True):
        return _get_transfer_model_(img_w=img_w, img_h=img_h, application_name=application_name,
                                    fine_tuning=fine_tuning)

    def get_question_model(self, question_len, wtoi):
        question_input = tf.keras.layers.Input(shape=question_len)
        lstm_model = tf.keras.layers.Embedding(len(wtoi) + 1, self.EMBEDDING_SIZE,
                                               input_length=question_len)(question_input)
        lstm_model = tf.keras.layers.LSTM(128, return_state=False)(lstm_model)
        lstm_model = tf.keras.Model(inputs=question_input, outputs=lstm_model)
        return lstm_model

    def get_model(self, question_len, wtoi, img_h, img_w, seed, fine_tuning=True, application_name="vgg16"):
        cnn_model = self.get_image_model(img_h, img_w, fine_tuning=fine_tuning, application_name=application_name)
        lstm_model = self.get_question_model(question_len, wtoi)

        model = tf.keras.layers.concatenate([cnn_model.output, lstm_model.output])
        model = tf.keras.layers.Dense(units=128)(model)
        model = tf.keras.layers.Dropout(0.2, seed=seed)(model)
        model = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(model)

        model = tf.keras.Model(inputs=[lstm_model.input, cnn_model.input], outputs=model)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


class ConvRecurrentNetworkExpFilters(object):
    EMBEDDING_SIZE = 50

    def get_image_model(self, img_h, img_w, start_f=32):
        model = tf.keras.models.Sequential()
        depth_max_pool = 3
        total_depth = 5
        k_init = "he_normal"

        model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3, 3), padding='same',
                                         input_shape=(img_h, img_w, 3),
                                         activation='elu', kernel_initializer=k_init))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        for i in range(1, total_depth - 1):
            model.add(
                tf.keras.layers.Conv2D(filters=start_f * (2 ** i), kernel_size=(3, 3), padding='same', activation='elu',
                                       kernel_initializer=k_init))
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
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        callbacks.append(es_callback)
