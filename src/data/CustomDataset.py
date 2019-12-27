from src.data.CustomDataGenerator import CustomTrainValidGenerator, CustomTestGenerator


def read_train_valid_data(train_questions_path: str, image_dir: str, img_h, img_w, tokenizer, split_seed,
                          batch_size=32):
    data_generator = CustomTrainValidGenerator(train_questions_path, image_dir, (img_h, img_w), tokenizer,
                                               split_seed=split_seed, valid_split=0.1)
    train_dataset = data_generator.get_train_dataset(batch_size)
    valid_dataset = data_generator.get_valid_dataset(batch_size)

    return train_dataset, valid_dataset, data_generator.get_train_samples(), data_generator.get_valid_samples()


def read_test_data(test_questions_path: str, image_dir, img_h, img_w, tokenizer, batch_size=32):
    data_generator = CustomTestGenerator(test_questions_path, image_dir, (img_h, img_w), tokenizer)
    test_dataset = data_generator.get_dataset(batch_size)
    return test_dataset, data_generator.questions_list
