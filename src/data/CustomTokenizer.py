import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_NUM_WORDS = 300
MAX_WORDS_IN_SENTENCE = 100


def get_question_shape():
    return MAX_WORDS_IN_SENTENCE


class CustomTokenizer(object):

    def __init__(self, train_file_path):
        with open(train_file_path, "r") as train_file:
            questions_list = json.load(train_file)['questions']
        all_train_questions = [question_dict['question'].replace("?", "") for question_dict in questions_list]
        all_train_questions.append(["sos", "eos"])

        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token=True)
        self.tokenizer.fit_on_texts(all_train_questions)

        self.questions_wtoi = self.tokenizer.word_index
        print('Total questions words:', len(self.questions_wtoi))

    def get_wtoi(self):
        return self.questions_wtoi

    def tokenize_and_pad(self, question):
        question_tokenized = self.tokenizer.texts_to_sequences([question])
        question_tokenized_and_pad = pad_sequences(question_tokenized, maxlen=MAX_WORDS_IN_SENTENCE)
        return question_tokenized_and_pad[0]

    def tokenize_and_pad_list(self, question_list):
        question_tokenized = self.tokenizer.texts_to_sequences(question_list)
        question_tokenized_and_pad = pad_sequences(question_tokenized, maxlen=MAX_WORDS_IN_SENTENCE)
        return question_tokenized_and_pad

