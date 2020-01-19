from src.data.CustomDataset import read_train_valid_data, read_test_data
import os
import math
import tensorflow as tf

from src.data.CustomTokenizer import CustomTokenizer
from src.model.models import TransferBidirectionalGRU
from src.submission.submission import predict_submissions

if __name__ == '__main__':
    seed = 241
    batch_size = 32
    img_h = 480
    img_w = 320
    epochs = 4
    tf.random.set_seed(seed)

    train_questions_path = "data/dataset_vqa/train_data.json"
    test_questions_path = "data/dataset_vqa/test_data.json"
    train_image_dir = "data/dataset_vqa/train"
    test_image_dir = "data/dataset_vqa/test"

    tokenizer = CustomTokenizer(train_questions_path)
    train_dataset, valid_dataset, train_samples, valid_samples = read_train_valid_data(train_questions_path,
                                                                                       train_image_dir,
                                                                                       img_h=img_h, img_w=img_w,
                                                                                       batch_size=batch_size,
                                                                                       tokenizer=tokenizer,
                                                                                       split_seed=seed)
    model = TransferBidirectionalGRU().get_model(100, tokenizer.get_wtoi(), img_h=img_h, img_w=img_w, seed=seed,
                                                 add_final_conv_layer=False, dropout_rate=0.25, n_units_question=128,
                                                 n_unit_dense=128, activation="relu", pooling="avg")
    model.summary()

    model.fit(x=train_dataset, epochs=epochs, steps_per_epoch=math.ceil(train_samples / batch_size),
              validation_data=valid_dataset, validation_steps=math.ceil(valid_samples / batch_size))

    test_dataset, questions_list = read_test_data(test_questions_path, test_image_dir, img_h=img_h, img_w=img_w,
                                                  tokenizer=tokenizer, batch_size=batch_size)
    predict_submissions(model, test_dataset, questions_list, batch_size)
