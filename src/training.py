'''
Yes
'''


import numpy as np
from numpy import ndarray
from feature_extraction import get_image_features
from helpers import extract_images, get_black_white


TRAINING_DIR = 'data/input/training_data'
VALIDATION_DIR = 'data/input/validation_data'


def get_training_data(verbose: bool = False) -> ndarray:
    '''
    Returns a `9990x12` set of `training_data`.

    Builds `training_data` from the files corresponding to training
    found in the 'train_and_valid' folder.

    Each row of the `training_data` corresponds to a handwritten digit.
        - row[0:9] contains the 9 `feature` values.
        - row[10] contains the `threshold` value.
        - row[11] contains the `class label`.
    '''

    if verbose:
        print(f'\n{"-" * 70}')
        print('Building Training Set...')

    training_data = []
    class_labels = []

    num_files = 10
    for i in range(num_files):
        filename = f'{TRAINING_DIR}/handwritten_samples_{i}.csv'
        images, labels = extract_images(file=filename)

        for label in labels:
            class_labels.append([label])

        if verbose:
            print(f'\tComputing Feature Values in ./{filename}')

        for image in images:
            binary_image = get_black_white(image)
            training_data.append(get_image_features(binary_image))

    # Create Column 11 of threshold values = -1.
    # 9,990 instead of 10,000 because we dropped the 10 labels.
    thresholds = np.full(shape=(9990, 1), fill_value=-1)

    # Concatenate Threshhold Column to TRAIN
    training_data = np.concatenate((training_data, thresholds), axis=1)

    # Concatenate Label Column to TRAIN
    training_data = np.concatenate((training_data, class_labels), axis=1)

    np.random.shuffle(training_data)

    assert isinstance(training_data, ndarray)

    return training_data


def get_validation_data(verbose: bool = False) -> ndarray:
    '''
    Returns a `2490x12` set of `validation_data`.

    Builds `validation_data` from the files corresponding to validation
    found in the 'train_and_valid' folder.
    '''

    if verbose:
        print(f'\n{"-" * 70}')
        print('Building Validation Set...')

    validation_data = []
    class_labels = []

    num_files = 10
    for i in range(num_files):
        filename = f'{VALIDATION_DIR}/handwritten_samples_{i}.csv'
        images, labels = extract_images(file=filename)

        for label in labels:
            class_labels.append([label])

        if verbose:
            print(f'\tComputing Feature Values in ./{filename}')

        for image in images:
            binary_image = get_black_white(image)
            validation_data.append(get_image_features(binary_image))

    # Create Column 11 of threshold values = -1.
    # 9,990 instead of 10,000 because we dropped the 10 labels.
    thresh_arr = np.full(shape=(2490, 1), fill_value=-1)

    # Concatenate Threshhold Column to TRAIN")
    validation_data = np.concatenate((validation_data, thresh_arr), axis=1)

    # Concatenate Labels Column to TRAIN")
    validation_data = np.concatenate((validation_data, class_labels), axis=1)

    # Randomly Permuting Rows of Training Data
    np.random.shuffle(validation_data)

    assert isinstance(validation_data, ndarray)

    return validation_data


def get_testing_data(filename: str, verbose: bool = False) -> ndarray:
    '''
    Returns a `len(file) x 11` set of `testing_data`.

    Builds `testing_data` from the files corresponding to testing
    found in the 'test' folder.
    '''

    assert isinstance(filename, str)

    if verbose:
        print(f'\n{"-" * 70}')
        print('Building Testing Set...')

    testing_data = []

    images, _ = extract_images(filename, has_label=False)

    for image in images:
        binary_image = get_black_white(image)
        testing_data.append(get_image_features(binary_image))

    thresholds = np.full(shape=(len(images), 1), fill_value=-1)
    testing_data = np.concatenate((testing_data, thresholds), axis=1)

    assert isinstance(testing_data, ndarray)

    return testing_data


def train(weight_vectors: ndarray, epochs: int) -> tuple[int, int, int]:
    '''
    Returns the weight vectors associated with the least amount of prediction
    errors.

    Prediction Method:
    argmax {wk, xk} for all k labels / weight vectors.

    Adjustment Method:
    weight_vectors[j] += np.multiply(η, x[0:10])
    weight_vectors[predict] -= np.multiply(η, x[0:10])

    For each image in the training dataset, make a prediction using that
    image's features and a current set of weights associated with those features.
    If the prediction is correct, our algorithm marks it as a successful
    prediction and moves on.
    If the prediction is incorrect, the algorithm will count it as an error and
    increase the weights associated with the actual digit's feature values
    while decreasing those of the incorrectly predicted digit.
    '''

    training_data = get_training_data(verbose=True)
    validation_data = get_validation_data(verbose=True)

    epoch_weights = []  # weights for each epoch.
    epoch_successes = []  # number of correct predictions for each epoch.
    epoch_errors = []  # number of incorrect predictions for each epoch.

    # pylint: disable-next=non-ascii-name
    η = float(0.08)  # Learning Constant, η (Eta).
    for _ in range(epochs):
        for row in training_data:
            class_label = int(row[10])

            features = row[0:10]
            logits = [np.dot(weights, features) for weights in weight_vectors]
            predicted_label = np.argmax(logits)

            if predicted_label != class_label:
                weight_vectors[class_label] += np.multiply(η, features)
                weight_vectors[predicted_label] -= np.multiply(η, features)

        num_correct, num_incorrect = validate(weight_vectors, validation_data)

        epoch_weights.append(weight_vectors.copy())
        epoch_successes.append(num_correct)
        epoch_errors.append(num_incorrect)

    return (epoch_weights[np.argmin(epoch_errors)],  # best weights
            np.sum(epoch_successes),  # total correct predictions
            np.sum(epoch_errors))  # total incorrect predictions


def validate(weight_vectors: ndarray, validation_data: ndarray) -> tuple[int, int]:
    '''
    Returns the number of successful and unsuccessful predictions made using
    the given `WEIGHT_VECTORS` on the given `VALID`ation set.
    '''

    assert isinstance(weight_vectors, ndarray)
    assert isinstance(validation_data, ndarray)

    num_successes, num_errors = 0, 0

    for row in validation_data:
        class_label = int(row[10])

        # res = []
        # for weights in WEIGHT_VECTORS:
        #     res.append(np.dot(weights, features := row[0:10]))
        features = row[0:10]
        logits = [np.dot(weights, features) for weights in weight_vectors]

        predicted_label = np.argmax(logits)

        if predicted_label == class_label:
            num_successes += 1
        else:
            num_errors += 1

    return num_successes, num_errors


def get_predictions(filename: str, weight_vectors: ndarray) -> list:
    '''
    Returns `predictions` for each image / handwritten digit in
    the given `FILE` using the given `WEIGHT_VECTORS`.
    '''

    assert isinstance(filename, str)
    assert isinstance(weight_vectors, ndarray)

    predictions = []

    testing_data = get_testing_data(filename, verbose=True)

    for row in testing_data:
        # I'm not sure if logits is the right term here.
        # logits + softmax = prediction ?
        features = row[0:10]
        logits = [np.dot(weights, features) for weights in weight_vectors]
        predictions.append(prediction := np.argmax(logits))
        # less readable version:
        # predictions.append(np.argmax(np.dot(weight_vectors, features)))

    assert isinstance(predictions, list)
    assert all(isinstance(prediction, np.int64) for prediction in predictions)

    return predictions
