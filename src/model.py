"""
A simple Single-Layer Muli-Class Perceptron Model.
"""

import numpy as np
from numpy import ndarray


class Model:
    """
    A simple Single-Layer Muli-Class Perceptron Model.
    """

    def __init__(self, weights: ndarray, verbose: bool = False) -> None:
        """
        Initializes the model with the given `weights`.
        """

        self.weights = weights
        self.verbose = verbose
        self.training_data = None
        self.validation_data = None

    def split(self, training_data: ndarray) -> None:
        """
        Splits the given `training_data` into `training_data` and `validation_data`.
        """

        num_rows = training_data.shape[0]
        num_training = int(0.8 * num_rows)  # 80% training, 20% validation

        self.training_data = training_data[:num_training]
        self.validation_data = training_data[num_training:]

    def fit(self, training_data: ndarray, epochs: int) -> tuple[int, int, int]:
        """
        Returns the weight vectors associated with the least amount of
        prediction errors.
        """

        if self.verbose:
            print(f"{'-' * 70}")
            print("Fitting Model...")

        # Set self.training_data and self.validation_data
        self.split(training_data)

        epoch_weights = []  # weights for each epoch
        epoch_errors = []  # number of incorrect predictions for each epoch

        eta = float(0.08)  # learning rate, η
        for epoch in range(epochs + 1):
            for row in self.training_data:
                class_label = int(row[10])

                features = row[0:10]
                logits = [np.dot(wv, features) for wv in self.weights]
                predicted_label = np.argmax(logits)

                if predicted_label != class_label:
                    self.weights[class_label] += np.multiply(eta, features)
                    self.weights[predicted_label] -= np.multiply(eta, features)

            num_correct, num_incorrect = self.validate(self.weights)
            epoch_weights.append(self.weights.copy())
            epoch_errors.append(num_incorrect)

            if self.verbose and (epoch + 1) % 10 == 0:
                accuracy = num_correct / (num_correct + num_incorrect)
                print(f"\tEpoch: {
                      epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}")

        self.weights = epoch_weights[np.argmin(epoch_errors)]

        if self.verbose:
            print(f"{'-' * 70}\n")

        return self.weights

    def validate(self, weights: ndarray) -> tuple[int, int]:
        """
        Returns the number of correct and incorrect predictions made
        using the given `weights`.
        """

        num_successes, num_errors = 0, 0
        for row in self.validation_data:
            label, features = int(row[10]), row[0:10]
            logits = [np.dot(w, features) for w in weights]
            prediction = np.argmax(logits)

            if prediction == label:
                num_successes += 1
            else:
                num_errors += 1

        return num_successes, num_errors

    def test(self, testing_data) -> list:
        """
        Tests the model on the validation data.
        Returns `predictions` for each image / handwritten digit in
        the given `FILE` using the given `WEIGHT_VECTORS`.
        """

        predictions = []  # predicted labels
        for row in testing_data:
            features = row[0:10]
            logits = [np.dot(weights, features) for weights in self.weights]
            predictions.append(np.argmax(logits))

        assert all(isinstance(prediction, np.int64)
                   for prediction in predictions)

        if self.verbose:
            print(f"{'-' * 70}")
            print(f"Predicted Labels:\n{predictions}")
            print(f"{'-' * 70}\n")

        return predictions
