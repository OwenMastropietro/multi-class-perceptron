"""
Training a simple Single-Layer Multi-Class Perceptron Model.
"""

import numpy as np
from numpy import ndarray, random
from dataloader import DataLoader
from model import Model

np.set_printoptions(edgeitems=10, linewidth=180)


NUM_CLASSES = 10
NUM_FEATURES = 10


def random_weights(shape=(10, 10)) -> ndarray:
    """
    Returns weights with the given shape having random values
    between -0.05 and 0.05.
    """

    return random.uniform(low=-0.05, high=0.05, size=shape)


def main():
    """
    Le Main
    """

    # Initialize the DataLoader.
    data_loader = DataLoader(verbose=True)

    # Load and split the training data.
    training_data = data_loader.training_data()

    # Initialize and train the Model.
    model = Model(weights=random_weights(shape=(NUM_CLASSES, NUM_FEATURES)),
                  verbose=True)
    model.fit(training_data, epochs=100)  # fit model wieghts to training data
    model.test(data_loader.testing_data())  # test model on testing data

    return 0


if __name__ == "__main__":
    main()
