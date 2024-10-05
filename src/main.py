"""
Training a simple Single-Layer Multi-Class Perceptron Model.
"""

import numpy as np
from numpy import ndarray, random
from dataloader import DataLoader
from model import Model

np.set_printoptions(edgeitems=10, linewidth=180)


def random_weights(shape=(10, 10)) -> ndarray:
    """
    Returns weights with the given `shape` having random values
    between -0.05 and 0.05.
    """

    return random.uniform(low=-0.05, high=0.05, size=shape)


def main():
    """
    Le Main
    """

    # Initialize the DataLoader
    data_loader = DataLoader(verbose=True)

    # Initialize the Model
    model = Model(weights=random_weights(shape=(10, 10)),  # 10 classes, 10 features
                  validation_data=data_loader.validation_data(),
                  testing_data=data_loader.testing_data())

    # Fit the Model wieghts to the data
    model.fit(training_data=data_loader.training_data(), epochs=100)

    # Test the Model
    model.test()

    return 0


if __name__ == "__main__":
    main()
