import os
from datetime import datetime
import numpy as np
import math


def create_csv(results, results_dir='./'):
    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:
        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(str(key) + ',' + str(value) + '\n')


def predict_submissions(model, test_dataset, questions_list, batch_size, result_dir="./", ):
    """
    Given a model, test dataset and questions_list, it save a csv file containing the predictions of the model for the
    Kaggle competition
    """
    predictions = model.predict(x=test_dataset, steps=math.ceil(len(questions_list) / batch_size), verbose=1)
    predicted_class = np.argmax(predictions, axis=1)

    question_id_list = [question_dict['question_id'] for question_dict in questions_list]

    results = dict(zip(question_id_list, predicted_class))
    create_csv(results, results_dir=result_dir)

    print("Wrote file csv")
