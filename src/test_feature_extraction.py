'''
Unit tests for feature_extraction.py
'''

import unittest
from ast import literal_eval
import numpy as np

from helpers import extract_images, get_black_white

from feature_extraction import \
    density, \
    horizontal_symmetry, \
    horizontal_intersections, \
    vertical_intersections, \
    number_of_loops, \
    horizontally_split_symmetry, \
    vertically_split_symmetry, \
    get_image_features


# pylint: disable=invalid-name
# pylint: disable=unused-variable


def generate_all_images() -> list[np.ndarray]:
    '''Returns a list of images.'''

    folder = 'input_files/training_data'

    images = []
    for digit in range(NUM_FILES := 10):
        FILENAME = f'{folder}/handwritten_samples_{digit}.csv'
        IMAGES, _ = extract_images(file=FILENAME, has_label=True)
        BINARY_IMAGES = [get_black_white(image) for image in IMAGES]
        for image in BINARY_IMAGES:
            images.append(image)

    assert isinstance(images, list)
    assert [isinstance(image, np.ndarray) for image in images]
    assert len(images) == 9990, \
        f'Expected 9990 images, got {len(images)}'

    return images


def load_expected(filename: str) -> list[float]:
    '''Loads expected values from a file'''

    FOLDER = 'output_files/feature_values'
    VALID_FILENAMES = [
        f'{FOLDER}/density.expected',
        f'{FOLDER}/horizontal_symmetry.expected',
        f'{FOLDER}/horizontal_intersections.expected',
        f'{FOLDER}/vertical_intersections.expected',
        f'{FOLDER}/number_of_loops.expected',
        f'{FOLDER}/horizontally_split_symmetry.expected',
        f'{FOLDER}/vertically_split_symmetry.expected',
    ]

    assert filename in VALID_FILENAMES, \
        f'Invalid filename: {filename}'

    with open(filename, 'r', encoding='utf-8') as f:
        if filename.endswith('horizontal_intersections.expected') or \
                filename.endswith('vertical_intersections.expected'):
            expected_values = [literal_eval(value) for value in f.readlines()]
        else:
            expected_values = [float(value) for value in f.readlines()]

    # asert assert assert
    assert isinstance(expected_values, list)
    if filename.endswith('horizontal_intersections.expected') or \
            filename.endswith('vertical_intersections.expected'):
        for value in expected_values:
            assert isinstance(value, tuple)
            assert len(value) == 2, f'Expected length 2, got {len(value)}'
            assert [isinstance(v, float) for v in value]
    else:
        assert [isinstance(value, float) for value in expected_values]
    assert len(expected_values) == 9990, \
        f'Expected 9990 values, got {len(expected_values)}'

    return expected_values


class TestFeatureExtraction(unittest.TestCase):
    '''Test class for feature_extraction.py'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._images = generate_all_images()

        FOLDER = 'output_files/feature_values'

        self._densities = load_expected(
            filename=f'{FOLDER}/density.expected')

        self._horizontal_symmetries = load_expected(
            filename=f'{FOLDER}/horizontal_symmetry.expected')

        self._vertical_intersections = load_expected(
            filename=f'{FOLDER}/vertical_intersections.expected')

        self._horizontal_intersections = load_expected(
            filename=f'{FOLDER}/horizontal_intersections.expected')

        self._number_of_loops = load_expected(
            filename=f'{FOLDER}/number_of_loops.expected')

        self._horizontally_split_symmetries = load_expected(
            filename=f'{FOLDER}/horizontally_split_symmetry.expected')

        self._vertically_split_symmetries = load_expected(
            filename=f'{FOLDER}/vertically_split_symmetry.expected')

    def test_density(self):
        '''Tests the density function'''

        # test edge cases
        self.assertEqual(density(np.zeros(shape=(3, 3))), 0.0)
        self.assertEqual(density(np.eye(3)), 0.3333333333333333)
        self.assertEqual(density(np.ones(shape=(3, 3))), 1.0)

        # test custom expected values
        for i, image in enumerate(self._images):
            self.assertEqual(density(image), self._densities[i])

    def test_horizontal_symmetry(self):
        '''Tests the horizontal_symmetry function'''

        # test edge cases
        IMAGE = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        IDENTITY = np.eye(3, dtype=np.uint8)
        self.assertEqual(horizontal_symmetry(np.zeros((3, 3), np.uint8)), 0.0)
        self.assertEqual(horizontal_symmetry(np.ones((3, 3), np.uint8)), 0.0)
        self.assertEqual(horizontal_symmetry(IMAGE), 0.0)
        self.assertEqual(horizontal_symmetry(IDENTITY), 0.4444444444444444)

        # test custom expected values
        for i, image in enumerate(self._images):
            self.assertEqual(horizontal_symmetry(image),
                             self._horizontal_symmetries[i])

    def test_vertical_intersections(self):
        '''Tests the vertical_intersections function'''

        # test edge cases
        image = np.zeros(shape=(3, 3), dtype=np.uint8)
        self.assertEqual(vertical_intersections(image), (0.0, 0.0))

        image = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(vertical_intersections(image), (1.0, 1.0))

        image = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        self.assertEqual(vertical_intersections(image), (1.0, 1.0))

        image = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
        self.assertEqual(vertical_intersections(image), (1.0, 1.0))

        image = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
        self.assertEqual(vertical_intersections(image), (1.0, 1.0))

        image = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
        self.assertEqual(vertical_intersections(image), (2.0, 2.0))

        image = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        self.assertEqual(vertical_intersections(image), (1.0, 1.0))

        image = np.ones(shape=(3, 3), dtype=np.uint8)
        self.assertEqual(vertical_intersections(image), (1.0, 1.0))

        image = np.eye(3, dtype=np.uint8)
        self.assertEqual(vertical_intersections(image), (1.0, 1.0))

        image = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(vertical_intersections(
            image), (1.0, 0.3333333333333333))

        # test custom expected values
        for i, image in enumerate(self._images):
            self.assertEqual(vertical_intersections(image),
                             self._vertical_intersections[i])

    def test_horizontal_intersections(self):
        '''Tests the horizontal_intersections function'''

        # test edge cases
        image = np.zeros(shape=(3, 3), dtype=np.uint8)
        self.assertEqual(horizontal_intersections(image), (0.0, 0.0))

        image = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(horizontal_intersections(image), (1.0, 1.0))

        image = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(horizontal_intersections(image), (1.0, 1.0))

        image = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.assertEqual(horizontal_intersections(image), (1.0, 1.0))

        image = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
        self.assertEqual(horizontal_intersections(image), (1.0, 1.0))

        image = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        self.assertEqual(horizontal_intersections(image), (2.0, 2.0))

        image = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        self.assertEqual(horizontal_intersections(image), (1.0, 1.0))

        image = np.ones(shape=(3, 3), dtype=np.uint8)
        self.assertEqual(horizontal_intersections(image), (1.0, 1.0))

        image = np.eye(3, dtype=np.uint8)
        self.assertEqual(horizontal_intersections(image), (1.0, 1.0))

        # test custom expected values
        for i, image in enumerate(self._images):
            self.assertEqual(horizontal_intersections(image),
                             self._horizontal_intersections[i])

    def test_number_of_loops(self):
        '''Tests the number_of_loops function'''

        # test edge cases
        self.assertEqual(number_of_loops(np.zeros((28, 28), np.uint8)), 0.0)

        self.assertEqual(number_of_loops(np.ones((28, 28), np.uint8)), -1.0,
                         'Expect -1.0 for all 1 since we assume a background.')

        # test custom expected values
        for digit, image in enumerate(self._images):
            self.assertEqual(number_of_loops(image),
                             self._number_of_loops[digit])

    def test_horizontally_split_symmetry(self):
        '''Tests the horizontally_split_symmetry function'''

        # test edge cases
        image = np.zeros(shape=(4, 4), dtype=np.uint8)
        self.assertEqual(horizontally_split_symmetry(image), 0.0)

        image = np.ones(shape=(4, 4), dtype=np.uint8)
        self.assertEqual(horizontally_split_symmetry(image), 0.0)

        image = np.eye(4, dtype=np.uint8)
        self.assertEqual(horizontally_split_symmetry(image), 0.5)

        image = np.array([[1, 0],
                          [0, 0]])
        self.assertEqual(horizontally_split_symmetry(image), 0.5)

        image = np.array([[0, 1],
                          [0, 0]])
        self.assertEqual(horizontally_split_symmetry(image), 0.5)

        image = np.array([[0, 0],
                          [1, 0]])
        self.assertEqual(horizontally_split_symmetry(image), 0.5)

        image = np.array([[0, 0],
                          [0, 1]])
        self.assertEqual(horizontally_split_symmetry(image), 0.5)

        image = np.array([[1, 0],
                          [0, 1]])
        self.assertEqual(horizontally_split_symmetry(image), 1.0)

        image = np.array([[0, 1],
                          [1, 0]])
        self.assertEqual(horizontally_split_symmetry(image), 1.0)

        image = np.array([[1, 1],
                          [0, 0]])
        self.assertEqual(horizontally_split_symmetry(image), 0.0)

        image = np.array([[0, 0],
                          [1, 1]])
        self.assertEqual(horizontally_split_symmetry(image), 0.0)

        image = np.array([[1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1]])
        self.assertEqual(horizontally_split_symmetry(image), 1.0)

        # test custom expected values
        for i, image in enumerate(self._images):
            self.assertEqual(horizontally_split_symmetry(image),
                             self._horizontally_split_symmetries[i])

    def test_vertically_split_symmetry(self):
        '''Tests the vertically_split_symmetry function'''

        # test edge cases
        image = np.zeros(shape=(4, 4), dtype=np.uint8)
        self.assertEqual(vertically_split_symmetry(image), 0.0)

        image = np.ones(shape=(4, 4), dtype=np.uint8)
        self.assertEqual(vertically_split_symmetry(image), 0.0)

        image = np.eye(4, dtype=np.uint8)
        self.assertEqual(vertically_split_symmetry(image), 0.5)

        image = np.array([[1, 0],
                          [0, 0]])
        self.assertEqual(vertically_split_symmetry(image), 0.5)

        image = np.array([[0, 1],
                          [0, 0]])
        self.assertEqual(vertically_split_symmetry(image), 0.5)

        image = np.array([[0, 0],
                          [1, 0]])
        self.assertEqual(vertically_split_symmetry(image), 0.5)

        image = np.array([[0, 0],
                          [0, 1]])
        self.assertEqual(vertically_split_symmetry(image), 0.5)

        image = np.array([[1, 0],
                          [0, 1]])
        self.assertEqual(vertically_split_symmetry(image), 1.0)

        image = np.array([[0, 1],
                          [1, 0]])
        self.assertEqual(vertically_split_symmetry(image), 1.0)

        image = np.array([[1, 0],
                          [1, 0]])
        self.assertEqual(vertically_split_symmetry(image), 0.0)

        image = np.array([[0, 1],
                          [0, 1]])
        self.assertEqual(vertically_split_symmetry(image), 0.0)

        image = np.array([[1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1]])
        self.assertEqual(vertically_split_symmetry(image), 1.0)

        # test custom expected values
        for i, image in enumerate(self._images):
            self.assertEqual(vertically_split_symmetry(image),
                             self._vertically_split_symmetries[i])

    def test_get_image_features(self):
        '''Tests the get_image_features function'''

if __name__ == '__main__':
    unittest.main()
