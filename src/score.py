import os
from typing import Tuple, List


def lines_to_dict(lines: List, as_set: bool = False) -> dict:
    """
    Reads a submission or label file into a dictionary.
    Each row mus contain one image name followed by 0-many images names, all comma separated.
    :param lines:
    :param as_set: If true, items in the dictionary are stored as sets, else as lists
    :return: A dict with first image name as key and the other as values.
    """
    pic_dict = {}
    for line in lines:
        line = line.lower()
        ids = line.split(',')
        ids = [i.strip() for i in ids]
        ids = [i for i in ids if len(i) > 0]
        if as_set:
            pic_dict[ids[0]] = set(ids[1:])
        else:
            pic_dict[ids[0]] = ids[1:]
    return pic_dict


def get_submission_scores(submission_lines: str, label_dir: str) -> Tuple[float, float]:
    """
    Computes the test and validation score of a submission
    :param submission_lines: The lines of a csv. First entry is the picture for which matching pictures are predicted,
        followed by relevant matches with the second entry being the most relevant match, etc.
    :param label_dir: The folder where the csv that contain the labels against which we compare are located.
    :return: score on test dataset, score on the valdation data set
    """
    submission_dict = lines_to_dict(submission_lines)
    test_file_path = os.path.join(label_dir, 'gdsc_test.csv')
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    test_dict = lines_to_dict(lines, as_set=True)
    val_file_path = os.path.join(label_dir, 'gdsc_validation.csv')
    with open(val_file_path, 'r') as f:
        lines = f.readlines()
    val_dict = lines_to_dict(lines, as_set=True)

    score_test = get_submission_score(submission_dict, test_dict)
    score_val = get_submission_score(submission_dict, val_dict)
    return score_test, score_val


def get_submission_score(predictions: dict, labels: dict) -> float:
    """
    Takes the predictions and labels as dictionaries and computes the score
    with the formula: 10*f1@1 + 5*f1@2 + 2*f1@3 + f1@20
    :param predictions: A dict of image -> predicted matches, sorted by relevance
    :param labels: A dict of image -> actual matches
    :return: The score
    """
    score = 0.0
    for image, images in labels.items():
        try:
            images_pred = predictions[image]
            if len(images_pred) != len(set(images_pred)):
                raise ValueError
            # ('Duplicate images in predictions.')
            is_match = [int(i in images) for i in images_pred]
            total_images = len(images)
            f1_1 = f1_at_n(is_match, total_images, 1)
            f1_2 = f1_at_n(is_match, total_images, 2)
            f1_3 = f1_at_n(is_match, total_images, 3)
            f1_20 = f1_at_n(is_match, total_images, 20)
            score += 10*f1_1 + 5*f1_2 + 2*f1_3 + f1_20
        except KeyError:
            continue
    return score


def f1_at_n(is_match, potential_matches, n):
    """
    Takes a boolean list denoting if the n-th entry of the predictions is an actual match
    and the number of potential matches, i.e. how many matches are at most possible and
    an integer n and computed the f1 score if one were to only consider the n most
    relevant matches
    :param is_match:
    :param potential_matches:
    :param n:
    :return:
    """
    if potential_matches == 0:
        return 0
    correct_prediction = float(sum(is_match[:n]))
    precision = correct_prediction / n
    recall = correct_prediction / potential_matches
    try:
        f1 = 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0
    return f1
