'''
Yes
'''


import math
import pandas as pd
import numpy as np
from numpy import random, ndarray
from feature_extraction import get_image_features


# pylint: disable=unused-variable
# pylint: disable=invalid-name


np.set_printoptions(edgeitems=10, linewidth=180)


##########################################################################
#                                                                        #
# START: Helper Functions                                                #
#                                                                        #
##########################################################################


def normalize(vector: ndarray) -> ndarray:
    '''
    Returns the true element-wise division of the given `vector`
    by its `magnitude`.
    '''

    assert isinstance(vector, ndarray)

    def magnitude(vector: ndarray) -> ndarray:
        '''Returns the magnitude of the given `vector`.'''

        return math.sqrt(np.sum(np.square(vector)))

    return np.divide(vector, magnitude(vector))


def extract_images(file: str, has_label: bool = True) -> tuple[list, list]:
    '''
    Returns the (999) `images` and `labels` contained in the given `file`.

    - Description:
        1. Opens the CSV file pointed to by <path> for a specific
        digit (0-9), using pd.read_csv to store the <data> into a dataframe
        structure where each row is an image, and the first column is the label
        for that row.
        2. Stores a list of <labels> by converting the first column in the
        dataframe to a list.
        3. Stores a list of <images> by dropping the first <labels> column in
        the dataframe, treating the remaining columns in each row as pixel
        values to be iterated through, converting each row of pixel values into
        a recognizable 28x28 image (of a handwritten digit).
    '''

    pixel_data = pd.read_csv(file)

    labels = []
    if has_label:
        class_label = pixel_data.columns.values[0]
        labels = pixel_data[class_label].values.tolist()
        pixel_data = pixel_data.drop(class_label, axis=1)

    images = []
    # pylint: disable-next=unused-variable
    for img_idx in range(NUM_ROWS := pixel_data.shape[0]):
        img_as_row = pixel_data.iloc[img_idx].to_numpy()
        img_as_grid = np.reshape(img_as_row, newshape=(28, 28))  # 28x28=784
        images.append(img_as_grid)

    return images, labels


def get_black_white(image: ndarray, threshold=128) -> ndarray:
    '''
    Returns a black and white version of the given `image`, forcing all
    pixels greater than the given `threshold` to be `1` (black), else
    `0` (white).
    '''

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


def get_random_weights(shape=(10, 10)) -> ndarray:
    '''
    Returns weights with the given `shape` having random values
    between -0.05 and 0.05.
    '''

    return random.uniform(low=-0.05, high=0.05, size=shape)


def print_image(image: ndarray) -> None:
    '''
    Prints the given `image` to the console.
    '''

    assert isinstance(image, ndarray)

    num_rows, num_cols = image.shape

    for row in range(num_rows):
        for col in range(num_cols):
            # if image[row][col] < 128:
            if image[row][col] == 0:
                print(' ', end='')
            else:
                print('X', end='')
        print()


def generate_output_files():
    '''
    Generates output files for extract_images
    '''
    input_folder = 'input_files/training_data'
    output_folder = 'output_files/training_data'

    for i in range(NUM_FILES := 10):
        INPUT_FILE = f'{input_folder}/handwritten_samples_{i}.csv'
        OUTPUT_FILE = f'{output_folder}/greyscale_{i}.txt'

        try:
            with open(
                    file=OUTPUT_FILE,
                    mode='w',
                    encoding='utf-8') as f:
                IMAGES, LABELS = extract_images(INPUT_FILE, has_label=True)
                for _, (image, label) in enumerate(zip(IMAGES, LABELS)):
                    f.write(f'\n{label}')
                    f.write(f'\n{image}')

        except IOError as e:
            print(f'Error writing to {OUTPUT_FILE}: {e}')


def get_writable_feature_values(feature_values: dict[str, tuple[str, float]]) -> str:
    '''Returns a string of feature values that can be written to a file'''

    assert isinstance(feature_values, dict)

    result = ''

    for _, value in feature_values.items():
        feature_name, feature_value = value

        if isinstance(feature_value, float):
            feature_value = f'{feature_value:.6f}'

        result += f'{feature_name}: {feature_value}\n'

    return f'{result}\n'


def get_feature_values() -> None:
    '''Exports feature values to a file'''

    INPUT_FOLDER = 'input_files/training_data'
    OUTPUT_FOLDER = 'output_files/training_data'

    for digit in range(NUM_FILES := 10):
        INPUT_FILE = f'{INPUT_FOLDER}/handwritten_samples_{digit}.csv'
        OUTPUT_FILE = f'{OUTPUT_FOLDER}/feature_values_{digit}.txt'

        IMAGES, _ = extract_images(file=INPUT_FILE, has_label=True)

        BINARY_IMAGES = [get_black_white(image) for image in IMAGES]

        try:
            with open(
                    file=OUTPUT_FILE,
                    mode='w',
                    encoding='utf-8') as f:
                for image in BINARY_IMAGES:
                    feature_values = get_image_features(image, as_dict=True)
                    f.write(get_writable_feature_values(feature_values))
                    # f.write(f'\n\n{get_image_features(image)}')

        except IOError as e:
            print(f'Error writing to {OUTPUT_FILE}: {e}')

##########################################################################
#                                                                        #
# END: Helper Functions                                                  #
#                                                                        #
##########################################################################


if __name__ == '__main__':
    extract_images(file='input_files/training_data/handwritten_samples_0.csv')
    extract_images(
        file='input_files/validation_data/handwritten_samples_0.csv')
    extract_images(
        file='input_files/testing_data/unlabeled_digits.csv', has_label=False)
