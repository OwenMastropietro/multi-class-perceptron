"""
A module for loading training, validation, and testing data.
"""

import pandas as pd
import numpy as np
from numpy import ndarray
from feature_extraction import get_image_features


TRAINING_DIR = 'data/input/training_data'
VALIDATION_DIR = 'data/input/validation_data'
TEST_DATA_DIR = 'data/input/testing_data'


def extract_images(file: str, has_label: bool = True) -> tuple[list, list]:
    """
    Returns a tuple of (images, labels) extracted from a CSV file.
    """

    pixel_data = pd.read_csv(file)

    labels = []
    if has_label:
        class_label = pixel_data.columns.values[0]
        labels = pixel_data[class_label].values.tolist()
        pixel_data = pixel_data.drop(class_label, axis=1)

    images = []
    for img_idx in range(pixel_data.shape[0]):
        img_as_row = pixel_data.iloc[img_idx].to_numpy()
        img_as_grid = np.reshape(img_as_row, newshape=(28, 28))  # 28x28=784
        images.append(img_as_grid)

    return images, labels


def get_black_white(image: ndarray, threshold=128) -> ndarray:
    """
    Returns a black and white version of the given `image`, forcing all
    pixels greater than the given `threshold` to be `1` (black), else
    `0` (white).
    """

    assert isinstance(image, ndarray)

    black, white = 0, 1

    # PRE-NUMPY DAYS
    # pixels = []
    # for row in range(28):
    #     for col in range(28):
    #         if (int(image[row][col]) > 128):  # lower cut off?
    #             pixels.append(1)  # white
    #         else:
    #             pixels.append(0)  # black
    # return np.reshape(pixels, (28, 28))

    # Convert to Black and White.
    binary_image = np.where(image < threshold, black, white)

    assert isinstance(binary_image, ndarray)

    return binary_image


class DataLoader:
    """
    A class that loads the training, validation, and testing data.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def get_data(self, folder: str) -> ndarray:
        """
        Returns data from the given folder.

        Each row corresponds to a handwritten digit such that:
        - row[0:n-1] contains the n-2 feature values.
        - row[n-1] contains the threshold value.
        - row[n] contains the class label.
        """

        if self.verbose:
            print(f"{'-' * 70}")
            print("Building Training Set...")

        class_labels = []
        data = []
        for i in range(10):  # digits 0-9
            filename = f"{folder}/handwritten_samples_{i}.csv"
            if self.verbose:
                print(f"\tComputing Feature Values in ./{filename}")

            images, labels = extract_images(file=filename)
            for image, label in zip(images, labels):
                class_labels.append([label])
                data.append(get_image_features(get_black_white(image)))

        # Create a column of threshold values = -1.
        thresholds = np.full(shape=(len(data), 1), fill_value=-1)

        # Concatenate the threshhold and label columns to the data.
        data = np.concatenate((data, thresholds), axis=1)
        data = np.concatenate((data, class_labels), axis=1)

        # Shuffle the rows of the training data.
        np.random.shuffle(data)

        if self.verbose:
            print(f"\nExtracted {len(data)} samples.")
            print(f"{'-' * 70}\n")

        return data

    def training_data(self) -> ndarray:
        """
        Returns samples of data for training.
        """

        return self.get_data(TRAINING_DIR)

    def validation_data(self) -> ndarray:
        """
        Returns samples of data for validation.
        """

        return self.get_data(VALIDATION_DIR)

    def testing_data(self) -> ndarray:
        """
        Builds `testing_data` from the files corresponding to testing
        found in the 'test' folder.
        """

        if self.verbose:
            print(f"{'-' * 70}")
            print("Building Testing Set...")

        filename = f"{TEST_DATA_DIR}/unlabeled_digits.csv"
        images, _ = extract_images(filename, has_label=False)

        data = [get_image_features(get_black_white(image))
                for image in images]

        thresholds = np.full(shape=(len(data), 1), fill_value=-1)
        data = np.concatenate((data, thresholds), axis=1)

        if self.verbose:
            print(f"\nExtracted {len(data)} samples.")
            print(f"{'-' * 70}\n")

        return data
