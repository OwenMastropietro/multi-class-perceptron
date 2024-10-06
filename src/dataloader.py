"""
A module for loading training, validation, and testing data.
"""

import csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numpy import ndarray
from feature_extraction import get_image_features


TRAINING_DIR = 'data/input/training_data'
TESTING_DIR = 'data/input/testing_data'
BLACK = 0
WHITE = 1


def extract_images(file: str, has_label: bool = True) -> tuple[list, list]:
    """
    Returns a tuple of (images, labels) extracted from a CSV file.
    """

    images = []
    labels = []

    with open(file, mode="r", encoding="us-ascii") as f:
        reader = csv.reader(f)
        for row in reader:
            if has_label:
                label = int(row[0])
                labels.append(label)
                row = row[1:]  # remove the label
            image = np.array(row, dtype=int).reshape(28, 28)
            images.append(image)

    return images, labels


def get_black_white(image: ndarray, threshold=128) -> ndarray:
    """
    Returns the image in black and white.

    All pixels less than the threshold are set to black (0),
    and all others are set to white (1).
    """

    return np.where(image < threshold, BLACK, WHITE)


class DataLoader:
    """
    A class that loads the training, validation, and testing data.
    """

    def __init__(self,
                 training_dir: str = TRAINING_DIR,
                 testing_dir: str = TESTING_DIR,
                 verbose: bool = False) -> None:
        self.training_dir = training_dir
        self.testing_dir = testing_dir
        self.verbose = verbose

    def process_image(self, image, label):
        """
        Returns a tuple of (features, label) for the given image.
        """

        features = get_image_features(get_black_white(image))

        return features, label

    def training_data(self) -> ndarray:
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
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = []  # futures for parallel processing
            for i in range(10):  # digits 0-9
                filename = f"{self.training_dir}/handwritten_samples_{i}.csv"
                if self.verbose:
                    print(f"\tComputing Feature Values in ./{filename}")

                images, labels = extract_images(file=filename)
                for image, label in zip(images, labels):
                    futures.append(
                        executor.submit(self.process_image, image, label))

            for future in futures:
                features, label = future.result()
                class_labels.append([label])
                data.append(features)

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

    def testing_data(self) -> ndarray:
        """
        Returns data from the given folder.

        Each row corresponds to a handwritten digit such that:
        - row[0:n] contains the n-1 feature values.
        - row[n] contains the threshold value.
        - no class label
        """

        if self.verbose:
            print(f"{'-' * 70}")
            print("Building Testing Set...")

        filename = f"{self.testing_dir}/unlabeled_digits.csv"
        images, _ = extract_images(filename, has_label=False)

        data = [get_image_features(get_black_white(image))
                for image in images]

        thresholds = np.full(shape=(len(data), 1), fill_value=-1)
        data = np.concatenate((data, thresholds), axis=1)

        if self.verbose:
            print(f"\nExtracted {len(data)} samples.")
            print(f"{'-' * 70}\n")

        return data
