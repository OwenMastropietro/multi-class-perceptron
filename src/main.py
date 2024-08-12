'''
Yes
'''


import numpy as np
from feature_extraction import get_image_features
from training import train, get_predictions
from helpers import get_black_white, get_random_weights, extract_images

np.set_printoptions(edgeitems=10, linewidth=180)


def display_stats(training_stats: tuple[int, int, int], epochs: int) -> None:
    '''
    Will display the following information:
    - Best Weights.
    - Total Unsuccessful Predictions.
    - Total Successful Predictions.
    - Percent Error of Predictions.
    '''

    weights, num_errors, num_successes = training_stats

    success_rate = round(
        number=100 * (num_successes / (num_successes + num_errors)),
        ndigits=2)

    error_rate = round(
        number=100 * (num_errors / (num_errors + num_successes)),
        ndigits=2)

    print(f'\n{"-" * 70}')
    print('Best Weights:')
    print(weights)

    print(f'\n{"-" * 70}')
    print(f'Relative Success (over {epochs} Epochs on the validation data):')
    print(f'- Success Rate: {success_rate}%.', end=' ')
    print(f'(from {num_successes:,} successful predictions)')

    print(f'- Error Rate: {error_rate}%.', end=' ')
    print(f'(from {num_errors:,} unsuccessful predictions)')


def display_info(file: str) -> None:
    '''
    Given a file containing 'images', this method will convert each image to a
    black and white representation and display the label, the black and white
    image, and the 9 feature values associated with it.
    '''

    images, labels = extract_images(file)
    for i, label in enumerate(labels):
        image = images[i]
        binary_image = get_black_white(image)
        features = get_image_features(binary_image)

        print('\n-----------------------------------------------------\n')
        print(f'Label: {label}')
        print('Black & White (binary) Image:')
        print(binary_image)
        print(f'Feature 1: Density: {features[0]}')
        print(f'Feature 2: Degree of Horizontal Symmetry: {features[1]}')
        print(f'Feature 3: Horizontal Intersections (MAX): {features[2]}')
        print(f'Feature 4: Horizontal Intersections (AVG): {features[3]}')
        print(f'Feature 5: Vertical Intersections (MAX): {features[4]}')
        print(f'Feature 6: Vertical Intersections (AVG): {features[5]}')
        print(f'Feature 7: Number of Loops: {features[6]}')
        print(f'Feature 8: Degree of Symmetry (horizontal): {features[7]}')
        print(f'Feature 9: Degree of Symmetry (vertical): {features[8]}')
        print('\n-----------------------------------------------------\n')


def main():
    '''Le Main'''

    # Create Weight Vectors for Testing
    trained_data = train(weight_vectors=get_random_weights(), epochs=100)
    display_stats(training_stats=trained_data, epochs=100)

    # Test Weights on Test Data
    trained_weights = trained_data[0]
    unlabeled_digits = 'data/input/testing_data/unlabeled_digits.csv'
    predicted_labels = get_predictions(unlabeled_digits, trained_weights)

    # Display Predicted Labels
    print(f'\n{"-" * 70}')
    print('Predicted Labels:')
    print(predicted_labels)

    return 0


if __name__ == '__main__':
    main()
