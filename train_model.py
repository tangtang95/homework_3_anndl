from src.data.CustomDataset import read_train_valid_data, read_test_data
import os
import math

from src.data.CustomTokenizer import CustomTokenizer
from src.model.models import ConvRecurrentNetwork
from src.submission.submission import predict_submissions
from src.utils.utils import get_seed

if __name__ == '__main__':
    seed = get_seed()
    batch_size = 32
    img_h = 480
    img_w = 320

    data_path = "data/"
    dataset_vqa_path = os.path.join(data_path, "dataset_vqa")
    train_questions_path = os.path.join(dataset_vqa_path, "train_data.json")
    train_image_dir = os.path.join(dataset_vqa_path, "train")
    tokenizer = CustomTokenizer(train_questions_path)
    train_dataset, valid_dataset, train_samples, valid_samples = read_train_valid_data(train_questions_path,
                                                                                       train_image_dir,
                                                                                       img_h=img_h, img_w=img_w,
                                                                                       batch_size=batch_size,
                                                                                       tokenizer=tokenizer,
                                                                                       split_seed=seed)
    model = ConvRecurrentNetwork().get_model(100, tokenizer.get_wtoi(), img_h=img_h, img_w=img_w, seed=seed)
    model.fit(x=train_dataset, epoch=5, steps_per_epoch=math.ceil(train_samples/batch_size),
              validation_data=valid_dataset, validation_steps=math.ceil(valid_samples/batch_size))

    test_questions_path = os.path.join(dataset_vqa_path, "test_data.json")
    test_image_dir = os.path.join(dataset_vqa_path, "test")
    test_dataset, questions_list = read_test_data(test_questions_path, test_image_dir, img_h=img_h, img_w=img_w,
                                                  tokenizer=tokenizer, batch_size=batch_size)
    predict_submissions(model, test_dataset, questions_list, batch_size)

