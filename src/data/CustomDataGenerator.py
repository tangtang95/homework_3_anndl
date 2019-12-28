import json
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data.CustomTokenizer import CustomTokenizer

LABEL_DICT = {
    '0': 0,
    '1': 1,
    '10': 2,
    '2': 3,
    '3': 4,
    '4': 5,
    '5': 6,
    '6': 7,
    '7': 8,
    '8': 9,
    '9': 10,
    'no': 11,
    'yes': 12,
}


def get_number_of_labels():
    return len(list(LABEL_DICT.keys()))


def decode_img(image_tensor: tf.Tensor, image_size: tf.shape, rescale=True):
    image_tensor = tf.io.decode_png(image_tensor, channels=3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    image_tensor = tf.image.resize(image_tensor, image_size)
    if not rescale:
        image_tensor = image_tensor * 255.0
    return image_tensor


class CustomTrainValidGenerator(object):

    def __init__(self, questions_file_path: str, image_directory: str,
                 target_image_size: tf.shape, tokenizer: CustomTokenizer, valid_split: float,
                 split_seed, rescale_image=True):
        self.tokenizer = tokenizer

        with open(questions_file_path, 'r') as f:
            self.questions_list = json.load(f)['questions']

        self.train_questions_list, self.valid_questions_list = train_test_split(self.questions_list, shuffle=True,
                                                                                test_size=valid_split,
                                                                                random_state=split_seed)
        self.target_image_size = target_image_size
        self.image_directory = image_directory
        self.rescale_image = rescale_image

    def _dataset(self, questions_list, batch_size, do_shuffle=False):

        def process_image_filenames(image_filename):
            image_filepath = tf.strings.unicode_encode(image_filename, output_encoding='UTF-8')
            image_tensor = tf.io.read_file(image_filepath)
            image_tensor = decode_img(image_tensor, self.target_image_size, rescale=self.rescale_image)

            del image_filepath

            return image_tensor

        question_string_list = []
        image_filename_list = []
        label_list = []
        for question in questions_list:
            question_string_list.append(question['question'])

            image_filepath = os.path.join(self.image_directory, question['image_filename'])
            image_filename_list.append(image_filepath)

            label_list.append(LABEL_DICT[question['answer']])

        question_tokenized_list = self.tokenizer.tokenize_and_pad_list(question_string_list)
        image_filename_list = tf.strings.unicode_decode(image_filename_list, input_encoding='UTF-8')
        label_list = tf.one_hot(label_list, depth=get_number_of_labels(), dtype=tf.int32)

        question_dataset = tf.data.Dataset.from_tensor_slices(question_tokenized_list)
        label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
        image_dataset = tf.data.Dataset.from_tensor_slices(image_filename_list)
        image_dataset = image_dataset.map(process_image_filenames, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip(((question_dataset, image_dataset), label_dataset))
        if do_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size*5)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        dataset = dataset.repeat()
        return dataset

    def get_train_dataset(self, batch_size):
        return self._dataset(self.train_questions_list, batch_size, do_shuffle=True)

    def get_valid_dataset(self, batch_size):
        return self._dataset(self.valid_questions_list, batch_size)

    def get_train_samples(self):
        return len(self.train_questions_list)

    def get_valid_samples(self):
        return len(self.valid_questions_list)


class CustomTestGenerator(object):
    def __init__(self, questions_file_path: str, image_directory: str,
                 target_image_size: tf.shape, tokenizer: CustomTokenizer, rescale_image=True):
        self.tokenizer = tokenizer

        with open(questions_file_path, 'r') as f:
            self.questions_list = json.load(f)['questions']

        self.target_image_size = target_image_size
        self.image_directory = image_directory
        self.rescale_image = rescale_image

    def _dataset(self, questions_list, batch_size):

        def process_image_filenames(image_filename):
            image_filepath = tf.strings.unicode_encode(image_filename, output_encoding='UTF-8')
            image_tensor = tf.io.read_file(image_filepath)
            image_tensor = decode_img(image_tensor, self.target_image_size, rescale=self.rescale_image)

            del image_filepath

            return image_tensor

        question_string_list = []
        image_filename_list = []
        for question in questions_list:
            question_string_list.append(question['question'])

            image_filepath = os.path.join(self.image_directory, question['image_filename'])
            image_filename_list.append(image_filepath)

        question_tokenized_list = self.tokenizer.tokenize_and_pad_list(question_string_list)
        image_filename_list = tf.strings.unicode_decode(image_filename_list, input_encoding='UTF-8')

        question_dataset = tf.data.Dataset.from_tensor_slices(question_tokenized_list)
        image_dataset = tf.data.Dataset.from_tensor_slices(image_filename_list)
        image_dataset = image_dataset.map(process_image_filenames, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip(((question_dataset, image_dataset), question_dataset))  # Trick to have as input 2 arrays
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        dataset = dataset.repeat()
        return dataset

    def get_number_of_samples(self):
        return len(self.questions_list)

    def get_dataset(self, batch_size):
        return self._dataset(self.questions_list, batch_size=batch_size)
