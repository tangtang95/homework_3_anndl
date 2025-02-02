import tensorflow as tf
import os
from datetime import datetime

from src.data.CustomDataGenerator import get_number_of_labels


# ------ UTILS ------
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


def _get_transfer_model_(img_h, img_w, application_name="vgg16", fine_tuning=False, pooling=None,
                         add_final_conv_layer=True):
    if application_name == "vgg16":
        model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                  weights='imagenet',
                                                  input_shape=(img_h, img_w, 3),
                                                  pooling=pooling)
    elif application_name == "resnet50v2":
        model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                                           input_shape=(img_h, img_w, 3),
                                                           pooling=pooling)
    elif application_name == "inceptionresnetv2":
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                            input_shape=(img_h, img_w, 3),
                                                                            pooling=pooling)
    elif application_name == "nasnetmobile":
        model = tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights='imagenet',
                                                          input_shape=(img_h, img_w, 3))
    elif application_name == "mobilenetv2":
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet',
                                                               input_shape=(img_h, img_w, 3)
                                                               , pooling=pooling)
    else:
        raise NotImplemented("Transfer from {} model is not implemented.".format(application_name))

    if not fine_tuning:
        model.trainable = False

    if add_final_conv_layer and pooling is None and application_name == "vgg16":
        new_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                                          input_shape=(img_h, img_w, 3),
                                          activation='relu')(model.layers[-1].output)
        final_pooling = tf.keras.layers.GlobalAveragePooling2D()(new_conv)
        model = tf.keras.Model(inputs=model.input, outputs=final_pooling)

    return model


class RelationNetwork(object):
    EMBEDDING_SIZE = 50

    def get_image_model(self, img_h, img_w):
        model = tf.keras.models.Sequential()
        depth_max_pool = 5
        total_depth = 5
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                                         input_shape=(img_h, img_w, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        for i in range(1, total_depth):
            model.add(
                tf.keras.layers.Conv2D(filters=32 * (2 ** i), kernel_size=(3, 3), padding='same', activation='relu'))
            if i <= depth_max_pool:
                model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        return model

    def get_question_model(self, question_len, wtoi):
        question_input = tf.keras.layers.Input(shape=question_len)
        lstm_model = tf.keras.layers.Embedding(len(wtoi) + 1, self.EMBEDDING_SIZE,
                                               input_length=question_len)(question_input)
        lstm_model = tf.keras.layers.LSTM(256, return_state=False)(lstm_model)
        lstm_model = tf.keras.Model(inputs=question_input, outputs=lstm_model)
        return lstm_model

    def get_model(self, question_len, wtoi, img_h, img_w, seed):
        lstm_model = self.get_question_model(question_len, wtoi)
        cnn_model = self.get_image_model(img_h, img_w)
        output_shape = cnn_model.output_shape
        follow_cnn_model = tf.keras.layers.Reshape(target_shape=(output_shape[1] * output_shape[2], output_shape[3]))(
            cnn_model.output)

        lambda_layers = []
        for i in range(output_shape[1] * output_shape[2]):
            lambda_layer = tf.keras.layers.Lambda(lambda x: x[:, i, :])(follow_cnn_model)
            lambda_layers.append(lambda_layer)

        concat_layers = []
        shared_dense_layer = tf.keras.layers.Dense(units=256, activation="relu")
        for i in range(output_shape[1] * output_shape[2] - 1):
            for j in range(i + 1, output_shape[1] * output_shape[2]):
                concat_layer = tf.keras.layers.concatenate([lambda_layers[i], lambda_layers[j], lstm_model.output])
                concat_layer = shared_dense_layer(concat_layer)
                concat_layers.append(concat_layer)
        follow_cnn_model = tf.keras.layers.add(concat_layers)
        model = tf.keras.layers.Dense(units=192, activation="relu")(follow_cnn_model)
        model = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(model)

        model = tf.keras.Model(inputs=[lstm_model.input, cnn_model.input], outputs=model)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


class ConvTransferStackedAttention(object):
    def __init__(self, embedding_size=30):
        self.EMBEDDING_SIZE = embedding_size

    def get_image_model(self, img_h, img_w, application_name="vgg16", fine_tuning=True):
        return _get_transfer_model_(img_w=img_w, img_h=img_h, application_name=application_name,
                                    fine_tuning=fine_tuning, pooling="None", add_final_conv_layer=False)

    def get_question_model(self, question_len, wtoi, cnn_model: tf.keras.Model, n_features):
        # ATTENTION MODEL
        question_input = tf.keras.layers.Input(shape=question_len)
        question_embedding = tf.keras.layers.Embedding(len(wtoi) + 1, self.EMBEDDING_SIZE,
                                                       input_length=question_len)(question_input)
        question_embedding = tf.keras.layers.LSTM(units=n_features, return_sequences=True)(question_embedding)
        cnn_layer_image = tf.keras.layers.Conv2D(filters=n_features, kernel_size=(1, 1), padding="same")
        image_features = cnn_layer_image(cnn_model.output)
        output_shape = cnn_model.output_shape
        image_features = tf.keras.layers.Reshape(target_shape=(output_shape[1] * output_shape[2], n_features))(
            image_features)
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(question_embedding)

        attention_vector = question_embedding
        for i in range(3):
            query_embedding_dense = tf.keras.layers.Dense(units=n_features, use_bias=True)(attention_vector)
            query_embedding_dense = tf.keras.layers.Reshape(target_shape=(question_len, 1,
                                                                          n_features))(query_embedding_dense)

            image_dense = tf.keras.layers.Dense(units=n_features, use_bias=False)(image_features)
            image_dense = tf.keras.layers.Reshape(target_shape=(1, output_shape[1] * output_shape[2],
                                                                n_features))(image_dense)

            query_image_dense = tf.keras.layers.add([query_embedding_dense, image_dense])
            query_image_dense = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(query_image_dense)
            scores = tf.keras.layers.Activation(activation="tanh")(query_image_dense)
            probabilities = tf.keras.layers.Dense(units=output_shape[1] * output_shape[2], use_bias=True,
                                                  activation="softmax")(scores)
            image_features_estimates = tf.keras.layers.dot([probabilities, image_features], axes=[-1, 1])
            attention_vector = tf.keras.layers.add([image_features_estimates, query_encoding])

        attention_vector = tf.keras.layers.GlobalAveragePooling1D()(attention_vector)
        query_layer = tf.keras.layers.Concatenate()([query_encoding, attention_vector])

        attention_model = tf.keras.Model(inputs=[question_input, cnn_model.input], outputs=query_layer)

        return attention_model

    def get_model(self, question_len, wtoi, img_h, img_w, seed, fine_tuning=False, application_name="vgg16",
                  n_features=512):
        cnn_model = self.get_image_model(img_h, img_w, fine_tuning=fine_tuning, application_name=application_name)
        attention_model = self.get_question_model(question_len, wtoi, cnn_model, n_features=n_features)

        # TOP NET
        top_net = tf.keras.layers.Dense(units=256)(attention_model.output)
        top_net = tf.keras.layers.Dropout(0.2, seed=seed)(top_net)
        top_net = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(top_net)

        model = tf.keras.Model(inputs=attention_model.input, outputs=top_net)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


class ConvLSTM_LSTM_Network(object):
    EMBEDDING_SIZE = 50

    def get_image_model(self, img_h, img_w, application_name="vgg16", fine_tuning=True):
        if application_name == "vgg16":
            model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                      weights='imagenet',
                                                      input_shape=(img_h, img_w, 3),
                                                      pooling="None")
        elif application_name == "resnet50v2":
            model: tf.keras.Model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                                                               input_shape=(img_h, img_w, 3),
                                                                               pooling="None")
        elif application_name == "inceptionresnetv2":
            model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                                input_shape=(img_h, img_w, 3),
                                                                                pooling="None")
        else:
            raise NotImplemented("Transfer from this model is not implemented.")

        if not fine_tuning:
            model.trainable = False
        return model

    def get_question_model(self, question_len, wtoi):
        question_input = tf.keras.layers.Input(shape=question_len)
        lstm_model = tf.keras.layers.Embedding(len(wtoi) + 1, self.EMBEDDING_SIZE,
                                               input_length=question_len)(question_input)
        lstm_model = tf.keras.layers.LSTM(128, return_sequences=True, stateful=False)(lstm_model)
        lstm_model = tf.keras.layers.LSTM(128, return_sequences=False, stateful=False)(lstm_model)
        lstm_model = tf.keras.Model(inputs=question_input, outputs=lstm_model)
        return lstm_model

    def get_model(self, question_len, wtoi, img_h, img_w, batch_size, seed, fine_tuning=True, application_name="vgg16"):
        cnn_model = self.get_image_model(img_h, img_w, fine_tuning=fine_tuning, application_name=application_name)
        output_shape = cnn_model.output_shape
        follow_cnn_model = tf.keras.layers.Reshape(target_shape=(output_shape[1] * output_shape[2], output_shape[3]))(
            cnn_model.output)
        follow_cnn_model = tf.keras.layers.LSTM(units=256, return_state=False)(follow_cnn_model)

        lstm_model = self.get_question_model(question_len, wtoi)

        model = tf.keras.layers.concatenate([follow_cnn_model, lstm_model.output])
        model = tf.keras.layers.Dense(units=256, activation="relu")(model)
        model = tf.keras.layers.Dropout(0.2, seed=seed)(model)
        model = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(model)

        model = tf.keras.Model(inputs=[lstm_model.input, cnn_model.input], outputs=model)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


class TransferBidirectionalGRU(object):

    def __init__(self, embedding_size=25):
        self.EMBEDDING_SIZE = embedding_size

    def get_image_model(self, img_h, img_w, application_name="vgg16", fine_tuning=True,
                        add_final_conv_layer=True, pooling="avg"):
        return _get_transfer_model_(img_w=img_w, img_h=img_h, application_name=application_name,
                                    fine_tuning=fine_tuning,
                                    add_final_conv_layer=add_final_conv_layer,
                                    pooling=pooling)

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

    def get_model(self, question_len, wtoi, img_h, img_w, seed, n_unit_dense=128, n_units_question=256,
                  application_name="vgg16", fine_tuning=False, dropout_rate=0.2, activation=None,
                  add_final_conv_layer=True, pooling=None):
        cnn_model = self.get_image_model(img_h, img_w, application_name=application_name, fine_tuning=fine_tuning,
                                         add_final_conv_layer=add_final_conv_layer,
                                         pooling=pooling)
        question_model = self.get_question_model(question_len, wtoi, n_units=n_units_question)

        model = tf.keras.layers.concatenate([cnn_model.output, question_model.output])
        model = tf.keras.layers.Dense(units=n_unit_dense, activation=activation)(
            model)  # Activation has been set after fix
        model = tf.keras.layers.Dropout(rate=dropout_rate, seed=seed)(model)
        model = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(model)

        model = tf.keras.Model(inputs=[question_model.input, cnn_model.input], outputs=model)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model


class ConvTransferAttention(object):
    def __init__(self, embedding_size=50):
        self.EMBEDDING_SIZE = embedding_size

    def get_image_model(self, img_h, img_w, application_name="vgg16", fine_tuning=True):
        return _get_transfer_model_(img_w=img_w, img_h=img_h, application_name=application_name,
                                    fine_tuning=fine_tuning, pooling="None")

    def get_question_model(self, question_len, wtoi, cnn_model: tf.keras.Model, n_filters_conv=100):
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
        image_features = cnn_layer_image(cnn_model.output)
        output_shape = cnn_model.output_shape
        image_features = tf.keras.layers.Reshape(target_shape=(output_shape[1] * output_shape[2], n_filters_conv))(
            image_features)
        query_value_attention_seq = tf.keras.layers.AdditiveAttention()([question_embedding, image_features])

        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
            question_embedding)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
            query_value_attention_seq)

        query_layer = tf.keras.layers.Concatenate()(
            [query_encoding, query_value_attention])

        attention_model = tf.keras.Model(inputs=[question_input, cnn_model.input], outputs=query_layer)

        return attention_model

    def get_model(self, question_len, wtoi, img_h, img_w, seed, fine_tuning=True, application_name="vgg16",
                  n_conv_filters=512):
        cnn_model = self.get_image_model(img_h, img_w, fine_tuning=fine_tuning, application_name=application_name)
        attention_model = self.get_question_model(question_len, wtoi, cnn_model, n_filters_conv=n_conv_filters)

        # TOP NET
        top_net = tf.keras.layers.Dense(units=256)(attention_model.output)
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
        model = tf.keras.layers.Dense(units=128, activation="relu")(model)
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
        model = tf.keras.layers.Dense(units=128, activation="relu")(model)
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
        model = tf.keras.layers.Dense(units=128, activation="relu")(model)
        model = tf.keras.layers.Dropout(0.2, seed=seed)(model)
        model = tf.keras.layers.Dense(units=get_number_of_labels(), activation="softmax")(model)

        model = tf.keras.Model(inputs=[lstm_model.input, cnn_model.input], outputs=model)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"], optimizer="adam")

        return model
